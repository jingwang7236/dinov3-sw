"""自研蒸馏 Swin Transformer 包装为 Open-CD backbone。

将蒸馏得到的 Swin-huge 模型作为特征提取器，直接提取 4 级层级特征
[p2, p3, p4, p5]（对应 /4, /8, /16, /32），经 1×1 卷积投影到统一通道数，
可直接被 ChangeDinoDecoder / ChangeDinoCrossAttnDecoder 消费。

支持三种训练模式，可通过 FreezeScheduleHook 在指定迭代次数切换:
  - frozen:          完全冻结 Swin 主干，仅训练投影层
  - unfreeze_last_n: 解冻最后 N 个 transformer block + norm 层
  - full_finetune:   全量微调

接口与 DINOv3AdapterBackbone 完全一致:
  forward(x) -> List[Tensor]
  set_freeze_mode(mode, unfreeze_last_n)
"""
import logging
import sys
from contextlib import nullcontext
from typing import List

import torch
import torch.nn as nn

from opencd.registry import MODELS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 将蒸馏代码目录加入 sys.path，使 ``from dinov3.models.swin_transformer_upsample
# import swin_huge`` 可被解析。
# ---------------------------------------------------------------------------
SWIN_DISTILL_CODE_DIR = (
    '/mnt/qh2-nas3/00-model/00-wrs/dinov3_distill/'
    'dinov3-swin-distill-7B-with-gram-ori'
)
if SWIN_DISTILL_CODE_DIR not in sys.path:
    sys.path.insert(0, SWIN_DISTILL_CODE_DIR)


def _load_swin_state_dict(ckpt_path):
    """加载蒸馏 Swin checkpoint，返回可直接 load_state_dict 的 dict。"""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict):
        for key in ('model', 'state_dict', 'student', 'teacher'):
            if key in ckpt and isinstance(ckpt[key], dict):
                logger.info(f"Using ckpt['{key}'] ({len(ckpt[key])} keys)")
                return ckpt[key]
        has_tensors = any(isinstance(v, torch.Tensor) for v in ckpt.values())
        if has_tensors:
            state = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
            logger.info(f'Using top-level dict as state_dict ({len(state)} keys)')
            return state
    return ckpt


@MODELS.register_module()
class SwinDistillBackbone(nn.Module):
    """蒸馏 Swin Transformer backbone。

    直接提取 Swin 的 4 级层级特征，经 1×1 卷积投影到 out_channels。
    支持冻结 / 部分微调 / 全量微调三种训练模式。

    Args:
        out_channels (int): 输出通道数，应与 decode_head 的 fpn_channels 一致。
        swin_weight (str): 蒸馏 Swin 权重路径 (.pt)。
        device (str | None): 加载 Swin 时使用的设备，None 时不迁移。
        swin_model (str): Swin 模型规模，默认 'swin_huge'。
            可选 'swin_huge' / 'swin_large' / 'swin_base' / 'swin_small' / 'swin_tiny'。
        freeze_mode (str): 初始冻结模式:
            - 'frozen': 完全冻结 Swin 主干，仅训练投影层 (默认)。
            - 'full_finetune': 全量微调，Swin 所有参数可训练。
            - 'unfreeze_last_n': 仅解冻最后 ``unfreeze_last_n`` 个 transformer
              block 及最终 norm 层。
        unfreeze_last_n (int): ``freeze_mode='unfreeze_last_n'`` 时解冻的 block 数。
        norm_cfg (dict): 投影层 norm 配置，默认 SyncBN。
        act_cfg (dict): 投影层激活函数配置，默认 SiLU。
        teacher_dim (int): 蒸馏目标维度 (feature_proj 输出)，默认 4096。
            仅影响 feature_proj 层构建，下游任务不使用该层。
        **kwargs: 透传给 swin 模型工厂的其他参数。

    Note:
        Swin-huge 架构: embed_dim=352, depths=[2,2,18,2],
        num_heads=[8,16,32,64], window_size=8, patch_size=4。
        4 级特征通道: [352, 704, 1408, 2816]。
    """

    _MODEL_FACTORIES = None

    @classmethod
    def _get_factories(cls):
        if cls._MODEL_FACTORIES is None:
            from dinov3.models.swin_transformer_upsample import (
                swin_huge, swin_large, swin_base, swin_small, swin_tiny,
            )
            cls._MODEL_FACTORIES = {
                'swin_huge': swin_huge,
                'swin_large': swin_large,
                'swin_base': swin_base,
                'swin_small': swin_small,
                'swin_tiny': swin_tiny,
            }
        return cls._MODEL_FACTORIES

    def __init__(
        self,
        out_channels: int = 128,
        swin_weight: str = '/mnt/qh2-nas3/00-model/00-wrs/zhejiang_earth_results/swin-distill/30999/swintransformer-huge-upsample.pt',
        device: str = 'cuda',
        swin_model: str = 'swin_huge',
        freeze_mode: str = 'frozen',
        unfreeze_last_n: int = 0,
        norm_cfg: dict = None,
        act_cfg: dict = None,
        teacher_dim: int = 4096,
        **kwargs,
    ):
        super().__init__()

        if norm_cfg is None:
            norm_cfg = dict(type='SyncBN', requires_grad=True)
        if act_cfg is None:
            act_cfg = dict(type='SiLU', inplace=True)

        factories = self._get_factories()
        if swin_model not in factories:
            raise ValueError(
                f'Unknown swin_model: {swin_model!r}, '
                f'expected one of {list(factories.keys())}'
            )

        # 构建 Swin 模型（不加预训练权重）
        # img_size=256 保证所有 stage 分辨率 >= window_size(8)，
        # 避免 SwinTransformerBlock 自动缩小 window_size 导致与 checkpoint 不匹配
        self.swin = factories[swin_model](
            img_size=256,
            patch_size=4,
            in_chans=3,
            num_classes=1000,
            ape=False,
            patch_norm=True,
            masked_im_modeling=False,
            teacher_dim=teacher_dim,
            **kwargs,
        )

        # 加载蒸馏权重
        logger.info(f'Loading distilled Swin weights from: {swin_weight}')
        state = _load_swin_state_dict(swin_weight)
        msg = self.swin.load_state_dict(state, strict=False)
        matched = len(state) - len(msg.unexpected_keys)
        logger.info(
            f'Loaded distilled Swin weights: matched={matched}, '
            f'missing={len(msg.missing_keys)}, '
            f'unexpected={len(msg.unexpected_keys)}'
        )
        if msg.missing_keys:
            logger.warning(f'Missing keys (first 10): {msg.missing_keys[:10]}')
        if msg.unexpected_keys:
            logger.warning(f'Unexpected keys (first 10): {msg.unexpected_keys[:10]}')

        # 重置 WindowAttention 的 relative_position_index buffer
        from dinov3.models.swin_transformer_upsample import WindowAttention
        for module in self.swin.modules():
            if isinstance(module, WindowAttention):
                module.reset_relative_position_index()

        self.swin.eval()
        if device is not None:
            self.swin = self.swin.to(device)

        # 各阶段通道维度: [352, 704, 1408, 2816] for swin_huge
        embed_dim = self.swin.embed_dim
        num_layers = self.swin.num_layers
        stage_dims = [embed_dim * (2 ** i) for i in range(num_layers)]

        # 简单投影: 1x1 Conv + Norm + Act (各 stage 独立)
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.proj = nn.ModuleList([
            self._build_proj(dim, out_channels)
            for dim in stage_dims
        ])

        # 统计 Swin block 总数 (用于 unfreeze_last_n)
        self._all_blocks = []
        for layer in self.swin.layers:
            self._all_blocks.extend(layer.blocks)

        # 初始冻结模式
        self._current_freeze_mode = None
        self._current_unfreeze_n = 0
        self.set_freeze_mode(freeze_mode, unfreeze_last_n)

    def _build_proj(self, in_dim, out_dim):
        """构建单级投影层: 1x1 Conv + Norm + Act。"""
        norm_type = self.norm_cfg.get('type', 'SyncBN')
        act_type = self.act_cfg.get('type', 'SiLU')
        inplace = self.act_cfg.get('inplace', True)

        if norm_type == 'SyncBN':
            norm = nn.SyncBatchNorm
        elif norm_type in ('BN', 'BN2d', 'BatchNorm2d'):
            norm = nn.BatchNorm2d
        else:
            norm = nn.BatchNorm2d

        if act_type == 'SiLU':
            act = lambda: nn.SiLU(inplace=inplace)
        elif act_type == 'ReLU':
            act = lambda: nn.ReLU(inplace=inplace)
        elif act_type == 'GELU':
            act = lambda: nn.GELU()
        else:
            act = lambda: nn.SiLU(inplace=inplace)

        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
            norm(out_dim),
            act(),
        )

    # ------------------------------------------------------------------
    # 冻结模式切换（兼容 FreezeScheduleHook）
    # ------------------------------------------------------------------
    def set_freeze_mode(self, mode: str = 'frozen', unfreeze_last_n: int = 0):
        """动态切换 Swin 主干的冻结模式。

        可在训练过程中通过 FreezeScheduleHook 调用，切换后 optimizer 自动
        更新新解冻的参数。

        Args:
            mode: 冻结模式:
                - 'frozen': 冻结 Swin 全部参数，仅训练投影层。
                - 'full_finetune': 解冻 Swin 全部参数。
                - 'unfreeze_last_n': 冻结全部后，解冻最后 N 个 transformer
                  block + 最终 norm 层 (norm / cls_norm)。
            unfreeze_last_n: mode='unfreeze_last_n' 时解冻的 block 数。
                Swin-huge 共 24 个 block (depths=[2,2,18,2])。
        """
        swin = self.swin

        if mode == 'frozen':
            swin.requires_grad_(False)
        elif mode == 'full_finetune':
            swin.requires_grad_(True)
        elif mode == 'unfreeze_last_n':
            swin.requires_grad_(False)
            n_blocks = len(self._all_blocks)
            start = max(0, n_blocks - unfreeze_last_n)
            for i in range(start, n_blocks):
                for p in self._all_blocks[i].parameters():
                    p.requires_grad = True
            # 解冻最终 norm 层
            for attr in ('norm', 'cls_norm'):
                norm = getattr(swin, attr, None)
                if norm is not None:
                    for p in norm.parameters():
                        p.requires_grad = True
        else:
            raise ValueError(
                f"Unknown freeze_mode: '{mode}', expected one of "
                f"['frozen', 'full_finetune', 'unfreeze_last_n']"
            )

        self._current_freeze_mode = mode
        self._current_unfreeze_n = unfreeze_last_n

        trainable = sum(p.numel() for p in swin.parameters() if p.requires_grad)
        total = sum(p.numel() for p in swin.parameters())
        logger.info(
            f'SwinDistillBackbone freeze_mode={mode!r}: '
            f'trainable {trainable:,} / {total:,} ({100*trainable/total:.1f}%)'
        )

    @property
    def _swin_frozen(self):
        return self._current_freeze_mode == 'frozen'

    # ------------------------------------------------------------------
    # forward: 提取 4 级多尺度特征
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """前向传播，返回 4 级多尺度特征列表。

        Args:
            x: [B, 3, H, W] 输入图像

        Returns:
            [p2, p3, p4, p5]: 4 个尺度的特征，通道数为 out_channels，
            分别对应 H/4, H/8, H/16, H/32
        """
        ctx = torch.no_grad() if self._swin_frozen else nullcontext()

        with ctx:
            # Patch embedding: [B, 3, H, W] -> [B, C, H/4, W/4]
            feat = self.swin.patch_embed(x)
            _, _, h, w = feat.shape
            x_tok = feat.flatten(2).transpose(1, 2)

            if self.swin.ape:
                x_tok = x_tok + self.swin.absolute_pos_embed
            x_tok = self.swin.pos_drop(x_tok)

            cur_h, cur_w = h, w
            raw_feats = []
            for layer in self.swin.layers:
                for blk in layer.blocks:
                    x_tok, _ = blk(x_tok)

                # [B, L, C] -> [B, C, H, W]
                B2, L, C = x_tok.shape
                feat_2d = x_tok.transpose(1, 2).reshape(
                    B2, C, cur_h, cur_w).contiguous()
                raw_feats.append(feat_2d)

                # PatchMerging 降采样
                if layer.downsample is not None:
                    x_tok = layer.downsample(x_tok)
                    cur_h, cur_w = cur_h // 2, cur_w // 2

        # 通道投影
        out = [proj(f) for proj, f in zip(self.proj, raw_feats)]
        return out


# 向后兼容别名
SwinDistillAdapterBackbone = SwinDistillBackbone
