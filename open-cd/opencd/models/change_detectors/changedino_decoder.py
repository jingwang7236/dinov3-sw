import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union

from opencd.models.blocks import TransformerBlock
from opencd.registry import MODELS

from opencd.models.losses import DICELoss, FocalLoss, LovaszSoftmaxLoss

class FuseGated(nn.Module):
    """Gated fusion module for multi-scale features."""

    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 1, bias=True),
            nn.Sigmoid()
        )
        self.mix = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if x1.shape[-2:] != x2.shape[-2:]:
            x1 = F.interpolate(
                x1, size=x2.shape[-2:], mode="bilinear", align_corners=False
            )
        g = self.gate(torch.cat([x1, x2], dim=1))
        fused = x2 + g * x1
        return self.mix(fused)


@MODELS.register_module()
class ChangeDinoDecoder(nn.Module):
    """ChangeDino Decoder.

    实现 ChangeDino 的多尺度差异特征提取与变化检测预测。
    兼容 SiamEncoderDecoder 父类的调用规范，提供:
    - forward(feats1, feats2): 多尺度特征提取
    - loss(feats1, feats2, data_samples, train_cfg): 计算训练损失
    - predict(feats1, feats2, batch_img_metas, test_cfg): 推理预测
    - _forward(feats1, feats2): 纯前向（用于 tensor mode）

    Args:
        fpn_channels (int): FPN 特征通道数 (default: 128)
        n_layers (List[int]): 各尺度 Transformer 块数量 (default: [1,1,1,1])
        num_classes (int): 分割类别数 (default: 2)
        align_corners (bool): 插值是否对齐角点
        aux_loss_weights (dict): 辅助损失权重 (default: p3=0.4, p4=0.3, p5=0.1)
        ignore_index (int): 损失计算中忽略的标签 (default: 255)
    """

    def __init__(
        self,
        fpn_channels: int = 128,
        n_layers: List[int] = [1, 1, 1, 1],
        num_classes: int = 2,
        align_corners: bool = False,
        aux_loss_weights: dict = None,
        ignore_index: int = 255,
        lovasz_weight: float = 0.0,
        **kwargs,
    ):
        super().__init__()

        # ========== 父类/框架要求的属性 ==========
        self.align_corners = align_corners
        self.num_classes = num_classes
        self.out_channels = num_classes
        self.ignore_index = ignore_index
        # Focal Loss (与官方一致: gamma=4.0)
        self.focal_loss = FocalLoss(
            alpha=0.25,
            gamma=4.0,
            ignore_index=ignore_index
        )
        # Dice Loss
        self.dice_loss = DICELoss(
            ignore_index=ignore_index
        )
        # Lovász Loss (可选, 默认关闭以保持向后兼容)
        self.lovasz_weight = lovasz_weight
        if lovasz_weight > 0:
            self.lovasz_loss = LovaszSoftmaxLoss(ignore_index=ignore_index)
        else:
            self.lovasz_loss = None
        # 各尺度损失权重
        self.aux_focal_weights = {
            'p2': 1.0,
            'p3': 1.0,
            'p4': 1.0,
            'p5': 1.0,
        }
        self.aux_dice_weights = {
            'p2': 0.5,
            'p3': 0.5,
            'p4': 0.5,
            'p5': 0.5,
        }
        self.aux_lovasz_weights = {
            'p2': 1.0,
            'p3': 0.5,
            'p4': 0.3,
            'p5': 0.1,
        }
        # ========== 模型参数 ==========
        self.fpn_channels = fpn_channels
        self.n_layers = n_layers

        # ========== 特征融合模块（自顶向下） ==========
        self.p5_to_p4 = FuseGated(fpn_channels)
        self.p4_to_p3 = FuseGated(fpn_channels)
        self.p3_to_p2 = FuseGated(fpn_channels)

        # ========== 各尺度的 Transformer 模块 ==========
        self.tb5 = self._build_transformer_blocks(
            fpn_channels, "CDA", n_layers[0], depth=3
        )
        self.tb4 = self._build_transformer_blocks(
            fpn_channels, "CDA", n_layers[1], depth=3
        )
        self.tb3 = self._build_transformer_blocks(
            fpn_channels, "OCDA", n_layers[2], depth=2,
            window_size=8, overlap_ratio=0.5
        )
        self.tb2 = self._build_transformer_blocks(
            fpn_channels, "OCDA", n_layers[3], depth=1,
            window_size=8, overlap_ratio=0.5
        )

        # ========== 预测头 ==========
        self.p5_head = nn.Conv2d(fpn_channels, num_classes, 1)
        self.p4_head = nn.Conv2d(fpn_channels, num_classes, 1)
        self.p3_head = nn.Conv2d(fpn_channels, num_classes, 1)
        self.p2_head = nn.Conv2d(fpn_channels, num_classes, 1)

    def _build_transformer_blocks(
        self, dim: int, attn_type: str, num_blocks: int,
        depth: int, **extra_kwargs
    ) -> nn.Module:
        """构建 Transformer 块序列。"""
        if num_blocks == 0:
            return nn.Identity()
        blocks = []
        for _ in range(num_blocks):
            block_kwargs = {
                'dim': dim,
                'spatial_attn_type': attn_type,
                'num_channel_heads': 8,
                'num_spatial_heads': 4,
                'depth': depth,
                'ffn_expansion_factor': 2,
                'bias': False,
                'LayerNorm_type': "BiasFree",
                **extra_kwargs
            }
            blocks.append(TransformerBlock(**block_kwargs))
        return nn.Sequential(*blocks)

    def _compute_diff(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        """计算双时相特征的差异。"""
        return torch.abs(t1 - t2)

    # ------------------------------------------------------------------
    # 核心前向：提取多尺度差异特征
    # ------------------------------------------------------------------
    def extract_feats(
        self, x1s: List[torch.Tensor], x2s: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """提取多尺度差异特征（ChangeDino 核心逻辑）。

        Args:
            x1s: 时相1的多尺度特征 [p2, p3, p4, p5]
            x2s: 时相2的多尺度特征 [p2, p3, p4, p5]

        Returns:
            多尺度差异特征 [feat_p2, feat_p3, feat_p4, feat_p5]
        """
        t1_p2, t1_p3, t1_p4, t1_p5 = x1s
        t2_p2, t2_p3, t2_p4, t2_p5 = x2s

        # 计算差异特征
        diff_p5 = self._compute_diff(t1_p5, t2_p5)
        diff_p4 = self._compute_diff(t1_p4, t2_p4)
        diff_p3 = self._compute_diff(t1_p3, t2_p3)
        diff_p2 = self._compute_diff(t1_p2, t2_p2)

        # 自顶向下融合处理
        feat_p5 = self.tb5(diff_p5)
        feat_p4 = self.p5_to_p4(feat_p5, diff_p4)
        feat_p4 = self.tb4(feat_p4)
        feat_p3 = self.p4_to_p3(feat_p4, diff_p3)
        feat_p3 = self.tb3(feat_p3)
        feat_p2 = self.p3_to_p2(feat_p3, diff_p2)
        feat_p2 = self.tb2(feat_p2)

        return [feat_p2, feat_p3, feat_p4, feat_p5]

    def _get_predictions(
        self, feats: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, ...]:
        """从多尺度特征生成各尺度的预测 logits。

        Args:
            feats: [feat_p2, feat_p3, feat_p4, feat_p5]

        Returns:
            (pred_p2, pred_p3, pred_p4, pred_p5) 各尺度 logits
        """
        pred_p2 = self.p2_head(feats[0])
        pred_p3 = self.p3_head(feats[1])
        pred_p4 = self.p4_head(feats[2])
        pred_p5 = self.p5_head(feats[3])
        return pred_p2, pred_p3, pred_p4, pred_p5

    def _parse_gt(self, data_samples: List) -> torch.Tensor:
        """从 data_samples 中解析 ground truth 标签。

        Args:
            data_samples: 数据样本列表

        Returns:
            gt_label_tensor: 形状 (B, H, W) 的 long tensor
        """
        gt_list = [d.gt_sem_seg.data for d in data_samples]
        gt_label_tensor = torch.cat(gt_list).long()
        if gt_label_tensor.ndim == 4 and gt_label_tensor.shape[1] == 1:
            gt_label_tensor = gt_label_tensor.squeeze(1)
        return gt_label_tensor

    # ------------------------------------------------------------------
    # forward: 对外暴露的通用前向接口
    # ------------------------------------------------------------------
    def forward(self, x1s, x2s, size=None):
        """前向传播，返回多尺度差异特征。

        Args:
            x1s: 时相1特征 (p2, p3, p4, p5)
            x2s: 时相2特征 (p2, p3, p4, p5)
            size: 保留参数

        Returns:
            [feat_p2, feat_p3, feat_p4, feat_p5]
        """
        return self.extract_feats(x1s, x2s)

    # ------------------------------------------------------------------
    # loss: 训练损失计算（符合父类 decode_head.loss 调用规范）
    # ------------------------------------------------------------------
    def loss(
        self,
        feats1: List[torch.Tensor],
        feats2: List[torch.Tensor],
        data_samples: List,
        train_cfg: Optional[dict] = None,
    ) -> dict:
        """计算训练损失。

        Args:
            feats1: 时相1的多尺度特征
            feats2: 时相2的多尺度特征
            data_samples: 数据样本列表，包含 ground truth
            train_cfg: 训练配置（可选）

        Returns:
            损失字典 {'loss_ce': ..., 'loss_aux_p3': ..., ...}
        """
        # 1. 提取差异特征
        feats = self.extract_feats(feats1, feats2)

        # 2. 各尺度预测
        pred_p2, pred_p3, pred_p4, pred_p5 = self._get_predictions(feats)

        # 3. 解析 GT
        gt_label_tensor = self._parse_gt(data_samples)
        target_size = gt_label_tensor.shape[-2:]

        # 4. 上采样预测到 GT 尺寸
        pred_p2 = F.interpolate(
            pred_p2, size=target_size, mode="bilinear",
            align_corners=self.align_corners)
        pred_p3 = F.interpolate(
            pred_p3, size=target_size, mode="bilinear",
            align_corners=self.align_corners)
        pred_p4 = F.interpolate(
            pred_p4, size=target_size, mode="bilinear",
            align_corners=self.align_corners)
        pred_p5 = F.interpolate(
            pred_p5, size=target_size, mode="bilinear",
            align_corners=self.align_corners)
        
        # 5. 计算损失
        scale_preds = {
            'p2': pred_p2,
            'p3': pred_p3,
            'p4': pred_p4,
            'p5': pred_p5,
        }
        total_focal = 0.0
        total_dice = 0.0

        # return losses
        for name, pred in scale_preds.items():
            focal_weight = self.aux_focal_weights.get(name, 0.0)
            dice_weight = self.aux_dice_weights.get(name, 0.0)
            
            if focal_weight > 0:
                total_focal += self.focal_loss(pred, gt_label_tensor) * focal_weight
            if dice_weight > 0:
                total_dice += self.dice_loss(pred, gt_label_tensor) * dice_weight
        
        # 总损失 = Focal + Dice
        losses = {}
        losses['loss_focal'] = total_focal
        losses['loss_dice'] = total_dice
        return losses
    # ------------------------------------------------------------------
    # predict: 推理预测（符合父类 decode_head.predict 调用规范）
    # ------------------------------------------------------------------
    def predict(
        self,
        feats1: List[torch.Tensor],
        feats2: List[torch.Tensor],
        batch_img_metas: List[dict],
        test_cfg: Optional[dict] = None,
    ) -> torch.Tensor:
        """推理预测，返回最高分辨率的分割 logits。

        Args:
            feats1: 时相1的多尺度特征
            feats2: 时相2的多尺度特征
            batch_img_metas: 图像元信息列表
            test_cfg: 测试配置（可选）

        Returns:
            seg_logits: 形状 (N, C, H, W) 的分割 logits
        """
        # 1. 提取差异特征
        feats = self.extract_feats(feats1, feats2)

        # 2. 最高分辨率预测
        pred_p2 = self.p2_head(feats[0])

        # 3. 上采样到原始图像尺寸
        img_shape = batch_img_metas[0]['ori_shape']
        pred_p2 = F.interpolate(
            pred_p2, size=img_shape, mode="bilinear",
            align_corners=self.align_corners
        )

        return pred_p2

    # ------------------------------------------------------------------
    # _forward: 纯前向（tensor mode，不计算损失）
    # ------------------------------------------------------------------
    def _forward(
        self,
        feats1: List[torch.Tensor],
        feats2: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, ...]:
        """纯前向传播，返回多尺度预测 logits（用于 tensor mode）。

        Args:
            feats1: 时相1的多尺度特征
            feats2: 时相2的多尺度特征

        Returns:
            (pred_p2, pred_p3, pred_p4, pred_p5) 各尺度 logits
        """
        feats = self.extract_feats(feats1, feats2)
        return self._get_predictions(feats)

    def loss_with_preds(
        self,
        feats1: List[torch.Tensor],
        feats2: List[torch.Tensor],
        data_samples: List,
        train_cfg: Optional[dict] = None,
    ) -> Tuple[dict, Tuple[torch.Tensor, ...]]:
        """计算训练损失并返回预测结果。
        
        与 loss 方法类似，但额外返回多尺度预测，供 refiner 使用。
        
        Returns:
            (loss_dict, preds): 损失字典和多尺度预测 (pred_p2, pred_p3, pred_p4, pred_p5)
        """
        # 1. 提取差异特征
        feats = self.extract_feats(feats1, feats2)

        # 2. 各尺度预测
        pred_p2, pred_p3, pred_p4, pred_p5 = self._get_predictions(feats)

        # 3. 解析 GT
        gt_label_tensor = self._parse_gt(data_samples)
        target_size = gt_label_tensor.shape[-2:]

        # 4. 上采样预测到 GT 尺寸
        pred_p2 = F.interpolate(
            pred_p2, size=target_size, mode="bilinear",
            align_corners=self.align_corners)
        pred_p3 = F.interpolate(
            pred_p3, size=target_size, mode="bilinear",
            align_corners=self.align_corners)
        pred_p4 = F.interpolate(
            pred_p4, size=target_size, mode="bilinear",
            align_corners=self.align_corners)
        pred_p5 = F.interpolate(
            pred_p5, size=target_size, mode="bilinear",
            align_corners=self.align_corners)

        scale_preds = {
            'p2': pred_p2,
            'p3': pred_p3,
            'p4': pred_p4,
            'p5': pred_p5,
        }
        
        total_focal = 0.0
        total_dice = 0.0
        
        for name, pred in scale_preds.items():
            focal_weight = self.aux_focal_weights.get(name, 0.0)
            dice_weight = self.aux_dice_weights.get(name, 0.0)
            
            if focal_weight > 0:
                total_focal += self.focal_loss(pred, gt_label_tensor) * focal_weight
            if dice_weight > 0:
                total_dice += self.dice_loss(pred, gt_label_tensor) * dice_weight
        
        losses = {}
        losses['loss_focal'] = total_focal
        losses['loss_dice'] = total_dice

        # Lovász loss (可选)
        if self.lovasz_loss is not None:
            total_lovasz = 0.0
            for name, pred in scale_preds.items():
                lw = self.aux_lovasz_weights.get(name, 0.0)
                if lw > 0:
                    total_lovasz += self.lovasz_loss(pred, gt_label_tensor) * lw
            losses['loss_lovasz'] = total_lovasz * self.lovasz_weight

        # 返回损失和所有尺度预测
        return losses, (pred_p2, pred_p3, pred_p4, pred_p5)


def _window_partition(x, ws):
    """(B, C, H, W) -> (B*nH*nW, ws*ws, C)，要求 H、W 能被 ws 整除。"""
    B, C, H, W = x.shape
    x = x.view(B, C, H // ws, ws, W // ws, ws)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()  # (B, nH, nW, ws, ws, C)
    return x.view(B * (H // ws) * (W // ws), ws * ws, C)


def _window_unpartition(x, B, H, W, ws):
    """(B*nH*nW, ws*ws, C) -> (B, C, H, W)。"""
    C = x.shape[-1]
    x = x.view(B, H // ws, W // ws, ws, ws, C)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()  # (B, C, nH, ws, nW, ws)
    return x.view(B, C, H, W)


class CrossAttnFusion(nn.Module):
    """双向窗口交叉注意力融合模块。

    输入同一尺度的双时相特征 (t1, t2)，输出用于变化检测的融合特征。
    采用对称交叉注意力 (t1 查询 t2、t2 查询 t1) 并以双向差作为输出，
    显式保留“变化”语义：
        out = Proj( XAttn(t1, t2) - XAttn(t2, t1) )

    Args:
        dim (int): 输入通道数。
        num_heads (int): 注意力头数。
        window_size (int): 窗口大小，需能整除各尺度特征的空间尺寸。
            window_size 较大时等价于全局交叉注意力。
        bias (bool): 卷积是否带偏置。
    """

    def __init__(self, dim, num_heads=4, window_size=8, bias=False):
        super().__init__()
        assert dim % num_heads == 0, "dim 必须能被 num_heads 整除"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.ws = window_size

        self.norm_q = nn.GroupNorm(1, dim)
        self.norm_kv = nn.GroupNorm(1, dim)
        self.to_q = nn.Conv2d(dim, dim, 1, bias=bias)
        self.to_kv = nn.Conv2d(dim, dim * 2, 1, bias=bias)
        self.proj = nn.Conv2d(dim, dim, 1, bias=bias)

    def _cross_attn(self, qf, kvf):
        """qf 作为 query，对 kvf 做 key/value 的交叉注意力。"""
        B, C, H, W = qf.shape
        ws = self.ws

        # 自适应 pad 到 ws 的整数倍，保证窗口切分合法
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        need_pad = (pad_h > 0) or (pad_w > 0)
        if need_pad:
            qf = F.pad(qf, (0, pad_w, 0, pad_h))
            kvf = F.pad(kvf, (0, pad_w, 0, pad_h))
        Hp, Wp = qf.shape[2], qf.shape[3]

        q = self.to_q(self.norm_q(qf))
        kv = self.to_kv(self.norm_kv(kvf))
        k, v = kv.chunk(2, dim=1)

        q = _window_partition(q, ws)   # (nw, ws*ws, C)
        k = _window_partition(k, ws)
        v = _window_partition(v, ws)
        nw = q.shape[0]
        N = ws * ws
        hd = self.num_heads

        q = q.view(nw, N, hd, self.head_dim).permute(0, 2, 1, 3) * self.scale
        k = k.view(nw, N, hd, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(nw, N, hd, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)   # (nw, hd, N, N)
        out = attn @ v                                      # (nw, hd, N, d)
        out = out.permute(0, 2, 1, 3).reshape(nw, N, C)
        out = _window_unpartition(out, B, Hp, Wp, ws)       # (B, C, Hp, Wp)

        if need_pad:
            out = out[:, :, :H, :W]
        return out

    def forward(self, t1, t2):
        a1 = self._cross_attn(t1, t2)   # t1 查询 t2
        a2 = self._cross_attn(t2, t1)   # t2 查询 t1
        return self.proj(a1 - a2)


@MODELS.register_module()
class ChangeDinoCrossAttnDecoder(ChangeDinoDecoder):
    """基于双向交叉注意力的 ChangeDino Decoder。

    继承 :class:`ChangeDinoDecoder` 的全部结构（FuseGated 自顶向下融合、
    TransformerBlock、各尺度预测头、损失函数等），仅将特征差分
    (``torch.abs(t1 - t2)``) 替换为 :class:`CrossAttnFusion`。

    其余参数与 ``ChangeDinoDecoder`` 完全一致，配置中只需将
    ``type='ChangeDinoDecoder'`` 改为 ``type='ChangeDinoCrossAttnDecoder'``，
    并可选地指定交叉注意力的 ``cross_num_heads`` / ``window_size``。

    额外 Args:
        cross_num_heads (int): 交叉注意力头数 (default: 4)
        window_size (int): 交叉注意力窗口大小 (default: 8)，需能整除
            各尺度特征空间尺寸。
    """

    def __init__(
        self,
        cross_num_heads: int = 4,
        window_size: int = 8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        dim = self.fpn_channels
        self.fuse5 = CrossAttnFusion(dim, cross_num_heads, window_size)
        self.fuse4 = CrossAttnFusion(dim, cross_num_heads, window_size)
        self.fuse3 = CrossAttnFusion(dim, cross_num_heads, window_size)
        self.fuse2 = CrossAttnFusion(dim, cross_num_heads, window_size)

    def extract_feats(
        self, x1s: List[torch.Tensor], x2s: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """用双向交叉注意力替代特征差分提取多尺度变化特征。

        与父类相比，仅 diff_p* 的计算方式不同；后续自顶向下融合与
        TransformerBlock 精修完全一致。
        """
        t1_p2, t1_p3, t1_p4, t1_p5 = x1s
        t2_p2, t2_p3, t2_p4, t2_p5 = x2s

        # 双向交叉注意力融合（替代 torch.abs(t1 - t2)）
        diff_p5 = self.fuse5(t1_p5, t2_p5)
        diff_p4 = self.fuse4(t1_p4, t2_p4)
        diff_p3 = self.fuse3(t1_p3, t2_p3)
        diff_p2 = self.fuse2(t1_p2, t2_p2)

        # 自顶向下融合（与父类一致）
        feat_p5 = self.tb5(diff_p5)
        feat_p4 = self.p5_to_p4(feat_p5, diff_p4)
        feat_p4 = self.tb4(feat_p4)
        feat_p3 = self.p4_to_p3(feat_p4, diff_p3)
        feat_p3 = self.tb3(feat_p3)
        feat_p2 = self.p3_to_p2(feat_p3, diff_p2)
        feat_p2 = self.tb2(feat_p2)

        return [feat_p2, feat_p3, feat_p4, feat_p5]