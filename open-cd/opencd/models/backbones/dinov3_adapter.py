# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
import math
import os
import re
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from opencd.registry import MODELS
from opencd.models.blocks.adapter import SepAdapterBlock, REPO_DIR, DINO_NAME
from opencd.models.blocks.ms_deform_attn import MSDeformAttn

logger = logging.getLogger(__name__)


def _is_official_dino_weights(weights):
    """判断 weights 路径是否为官方 DINOv3 权重（文件名含 -XXXXXXXX.pth 哈希）。"""
    if not isinstance(weights, str):
        return True
    pattern = r"-(.{8})\.pth$"
    return re.search(pattern, os.path.basename(weights)) is not None


def _clean_self_state_dict(state):
    """参考 load_checkpoint_forward.py 清理自研 checkpoint 的 key 前缀。"""
    cleaned = OrderedDict()
    for k, v in state.items():
        new_k = k
        for prefix in [
            "model.student.",
            "model.teacher.",
            "module.student.",
            "module.teacher.",
            "student.",
            "teacher.",
            "model.",
            "module.",
            "backbone.",
        ]:
            if new_k.startswith(prefix):
                new_k = new_k[len(prefix):]
                break
        new_k = new_k.replace("_orig_mod.", "")
        new_k = new_k.replace("_checkpoint_wrapped_module.", "")
        cleaned[new_k] = v
    return cleaned


def _extract_self_dino_state_dict(ckpt):
    """从自研 checkpoint 中提取 ViT 主干的 state_dict。

    支持多种顶层结构：ckpt['model'/'state_dict'/'student'/'teacher']，
    或顶层直接为 tensor dict。
    """
    if isinstance(ckpt, dict):
        for key in ("model", "state_dict", "student", "teacher"):
            if key in ckpt and isinstance(ckpt[key], dict):
                logger.info(f"Using ckpt['{key}'] ({len(ckpt[key])} keys)")
                return ckpt[key]
        # 顶层直接为 state_dict（过滤非 tensor）
        has_tensors = any(isinstance(v, torch.Tensor) for v in ckpt.values())
        if has_tensors:
            state = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
            logger.info(f"Using top-level dict as state_dict ({len(state)} tensor keys)")
            return state
    # 已经是 state_dict
    return ckpt


def load_self_dino_weights(model, ckpt_path):
    """加载自研 DINOv3 预训练权重到 DinoVisionTransformer 模型。

    采用 strict=False 加载，兼容 student ModuleDict 中 backbone.* 等 key。
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw_state = _extract_self_dino_state_dict(ckpt)
    cleaned = _clean_self_state_dict(raw_state)
    return load_self_dino_weights_from_state(model, cleaned)


def load_self_dino_weights_from_state(model, cleaned_state):
    """将已清理的 state_dict 加载到 DinoVisionTransformer 模型（strict=False）。"""
    msg = model.load_state_dict(cleaned_state, strict=False)
    matched = len(cleaned_state) - len(msg.unexpected_keys)
    logger.info(
        f"Loaded self-trained DINOv3 weights: matched={matched}, "
        f"missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}"
    )
    if msg.missing_keys:
        logger.warning(f"Missing keys (first 10): {msg.missing_keys[:10]}")
    if msg.unexpected_keys:
        logger.warning(f"Unexpected keys (first 10): {msg.unexpected_keys[:10]}")
    return model


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def get_reference_points(spatial_shapes, device):
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
        )
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None]
    return reference_points


def deform_inputs(x, patch_size):
    bs, c, h, w = x.shape
    spatial_shapes = torch.as_tensor(
        [(h // 8, w // 8), (h // 16, w // 16), (h // 32, w // 32)], dtype=torch.long, device=x.device
    )
    level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // patch_size, w // patch_size)], x.device)
    deform_inputs1 = [reference_points, spatial_shapes, level_start_index]

    spatial_shapes = torch.as_tensor([(h // patch_size, w // patch_size)], dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // 8, w // 8), (h // 16, w // 16), (h // 32, w // 32)], x.device)
    deform_inputs2 = [reference_points, spatial_shapes, level_start_index]

    return deform_inputs1, deform_inputs2


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        n = N // 21
        x1 = x[:, 0 : 16 * n, :].transpose(1, 2).view(B, C, H * 2, W * 2).contiguous()
        x2 = x[:, 16 * n : 20 * n, :].transpose(1, 2).view(B, C, H, W).contiguous()
        x3 = x[:, 20 * n :, :].transpose(1, 2).view(B, C, H // 2, W // 2).contiguous()
        x1 = self.dwconv(x1).flatten(2).transpose(1, 2)
        x2 = self.dwconv(x2).flatten(2).transpose(1, 2)
        x3 = self.dwconv(x3).flatten(2).transpose(1, 2)
        x = torch.cat([x1, x2, x3], dim=1)
        return x


class Extractor(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=6,
        n_points=4,
        n_levels=1,
        deform_ratio=1.0,
        with_cffn=True,
        cffn_ratio=0.25,
        drop=0.0,
        drop_path=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        with_cp=False,
    ):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(
            d_model=dim, n_levels=n_levels, n_heads=num_heads, n_points=n_points, ratio=deform_ratio
        )
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        if with_cffn:
            self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index, H, W):
        def _inner_forward(query, feat):
            attn = self.attn(
                self.query_norm(query), reference_points, self.feat_norm(feat), spatial_shapes, level_start_index, None
            )
            query = query + attn

            if self.with_cffn:
                query = query + self.drop_path(self.ffn(self.ffn_norm(query), H, W))
            return query

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)

        return query


class InteractionBlockWithCls(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=6,
        n_points=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        drop=0.0,
        drop_path=0.0,
        with_cffn=True,
        cffn_ratio=0.25,
        init_values=0.0,
        deform_ratio=1.0,
        extra_extractor=False,
        with_cp=False,
    ):
        super().__init__()
        self.extractor = Extractor(
            dim=dim,
            n_levels=1,
            num_heads=num_heads,
            n_points=n_points,
            norm_layer=norm_layer,
            deform_ratio=deform_ratio,
            with_cffn=with_cffn,
            cffn_ratio=cffn_ratio,
            drop=drop,
            drop_path=drop_path,
            with_cp=with_cp,
        )
        if extra_extractor:
            self.extra_extractors = nn.Sequential(
                *[
                    Extractor(
                        dim=dim,
                        num_heads=num_heads,
                        n_points=n_points,
                        norm_layer=norm_layer,
                        with_cffn=with_cffn,
                        cffn_ratio=cffn_ratio,
                        deform_ratio=deform_ratio,
                        drop=drop,
                        drop_path=drop_path,
                        with_cp=with_cp,
                    )
                    for _ in range(2)
                ]
            )
        else:
            self.extra_extractors = None

    def forward(self, x, c, cls, deform_inputs1, deform_inputs2, H_c, W_c, H_toks, W_toks):
        c = self.extractor(
            query=c,
            reference_points=deform_inputs2[0],
            feat=x,
            spatial_shapes=deform_inputs2[1],
            level_start_index=deform_inputs2[2],
            H=H_c,
            W=W_c,
        )
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(
                    query=c,
                    reference_points=deform_inputs2[0],
                    feat=x,
                    spatial_shapes=deform_inputs2[1],
                    level_start_index=deform_inputs2[2],
                    H=H_c,
                    W=W_c,
                )
        return x, c, cls


class SpatialPriorModule(nn.Module):
    def __init__(self, inplanes=64, embed_dim=384, with_cp=False):
        super().__init__()
        self.with_cp = with_cp

        self.stem = nn.Sequential(
            *[
                nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(inplanes),
                nn.ReLU(inplace=True),
                nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                nn.SyncBatchNorm(inplanes),
                nn.ReLU(inplace=True),
                nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                nn.SyncBatchNorm(inplanes),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ]
        )
        self.conv2 = nn.Sequential(
            *[
                nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(2 * inplanes),
                nn.ReLU(inplace=True),
            ]
        )
        self.conv3 = nn.Sequential(
            *[
                nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(4 * inplanes),
                nn.ReLU(inplace=True),
            ]
        )
        self.conv4 = nn.Sequential(
            *[
                nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(4 * inplanes),
                nn.ReLU(inplace=True),
            ]
        )
        self.fc1 = nn.Conv2d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc4 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        def _inner_forward(x):
            c1 = self.stem(x)
            c2 = self.conv2(c1)
            c3 = self.conv3(c2)
            c4 = self.conv4(c3)
            c1 = self.fc1(c1)
            c2 = self.fc2(c2)
            c3 = self.fc3(c3)
            c4 = self.fc4(c4)

            bs, dim, _, _ = c1.shape
            # c1 = c1.view(bs, dim, -1).transpose(1, 2)  # 4s
            c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s
            c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s
            c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s

            return c1, c2, c3, c4

        if self.with_cp and x.requires_grad:
            outs = cp.checkpoint(_inner_forward, x)
        else:
            outs = _inner_forward(x)
        return outs


class DINOv3_Adapter(nn.Module):
    def __init__(
        self,
        backbone,
        interaction_indexes=[9, 19, 29, 39],
        pretrain_size=512,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=16,
        drop_path_rate=0.3,
        init_values=0.0,
        with_cffn=True,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        add_vit_feature=True,
        use_extra_extractor=True,
        with_cp=True,
        backbone_requires_grad=False,
    ):
        super(DINOv3_Adapter, self).__init__()
        self.backbone = backbone
        self.backbone_requires_grad = backbone_requires_grad
        if not backbone_requires_grad:
            self.backbone.requires_grad_(False)

        self.pretrain_size = (pretrain_size, pretrain_size)
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        embed_dim = self.backbone.embed_dim
        self.patch_size = self.backbone.patch_size
        print("embed dim", embed_dim)
        print("interaction_indexes", self.interaction_indexes)
        print("patch_size", self.patch_size)

        block_fn = InteractionBlockWithCls
        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=embed_dim, with_cp=False)
        self.interactions = nn.Sequential(
            *[
                block_fn(
                    dim=embed_dim,
                    num_heads=deform_num_heads,
                    n_points=n_points,
                    init_values=init_values,
                    drop_path=drop_path_rate,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    with_cffn=with_cffn,
                    cffn_ratio=cffn_ratio,
                    deform_ratio=deform_ratio,
                    extra_extractor=(
                        (True if i == len(self.interaction_indexes) - 1 else False) and use_extra_extractor
                    ),
                    with_cp=with_cp,
                )
                for i in range(len(self.interaction_indexes))
            ]
        )
        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.norm1 = nn.SyncBatchNorm(embed_dim)
        self.norm2 = nn.SyncBatchNorm(embed_dim)
        self.norm3 = nn.SyncBatchNorm(embed_dim)
        self.norm4 = nn.SyncBatchNorm(embed_dim)

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        torch.nn.init.normal_(self.level_embed)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(
            1, self.pretrain_size[0] // self.patch_size, self.pretrain_size[1] // self.patch_size, -1
        ).permute(0, 3, 1, 2)
        pos_embed = (
            F.interpolate(pos_embed, size=(H, W), mode="bicubic", align_corners=False)
            .reshape(1, -1, H * W)
            .permute(0, 2, 1)
        )
        return pos_embed

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward(self, x):
        deform_inputs1, deform_inputs2 = deform_inputs(x, self.patch_size)

        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)

        c = torch.cat([c2, c3, c4], dim=1)

        # Code for matching with oss
        H_c, W_c = x.shape[2] // 16, x.shape[3] // 16
        H_toks, W_toks = x.shape[2] // self.patch_size, x.shape[3] // self.patch_size
        bs, C, h, w = x.shape

        with torch.autocast("cuda", torch.bfloat16):
            if self.backbone_requires_grad:
                all_layers = self.backbone.get_intermediate_layers(
                    x, n=self.interaction_indexes, return_class_token=True
                )
            else:
                with torch.no_grad():
                    all_layers = self.backbone.get_intermediate_layers(
                        x, n=self.interaction_indexes, return_class_token=True
                    )

        x_for_shape, _ = all_layers[0]
        bs, _, dim = x_for_shape.shape
        del x_for_shape

        outs = list()
        for i, layer in enumerate(self.interactions):
            x, cls = all_layers[i]
            _, c, _ = layer(
                x,
                c,
                cls,
                deform_inputs1,
                deform_inputs2,
                H_c,
                W_c,
                H_toks,
                W_toks,
            )
            outs.append(x.transpose(1, 2).view(bs, dim, H_toks, W_toks).contiguous())

        # Split & Reshape
        c2 = c[:, 0 : c2.size(1), :]
        c3 = c[:, c2.size(1) : c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1) :, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H_c * 2, W_c * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H_c, W_c).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H_c // 2, W_c // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x1, x2, x3, x4 = outs

            x1 = F.interpolate(x1, size=(4 * H_c, 4 * W_c), mode="bilinear", align_corners=False)
            x2 = F.interpolate(x2, size=(2 * H_c, 2 * W_c), mode="bilinear", align_corners=False)
            x3 = F.interpolate(x3, size=(1 * H_c, 1 * W_c), mode="bilinear", align_corners=False)
            x4 = F.interpolate(x4, size=(H_c // 2, W_c // 2), mode="bilinear", align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)

        return {"1": f1, "2": f2, "3": f3, "4": f4}


@MODELS.register_module()
class DINOv3AdapterBackbone(nn.Module):
    """将 DINOv3_Adapter 包装为 Open-CD backbone。

    输出与 ChangeDinoEncoderOnlyDino 兼容：返回 4 个尺度的特征列表
    [p2, p3, p4, p5]（对应 /4, /8, /16, /32），每个特征通道数为
    out_channels，可直接被 ChangeDinoDecoder 消费。

    Args:
        out_channels (int): 输出通道数，应与 decode_head 的 fpn_channels 一致。
        dino_weight (str): DINOv3 预训练权重路径。
        device (str): 加载 ViT 时使用的设备。
        extract_ids (List[int]): 交互的 ViT 层索引，同时作为 DINOv3_Adapter
            的 interaction_indexes。ViT-L 共 24 层，默认 [5, 11, 17, 23]。
        freeze_mode (str): ViT 主干冻结模式:
            - 'frozen': 完全冻结 ViT 主干，仅训练 adapter / 投影层 (默认)。
            - 'full_finetune': 全量微调，ViT 主干所有参数均可训练。
            - 'unfreeze_last_n': 仅解冻最后 ``unfreeze_last_n`` 层 transformer
              block + 最终 norm 层。
        unfreeze_last_n (int): 当 ``freeze_mode='unfreeze_last_n'`` 时指定
            解冻的 block 数量。
        pretrain_size (int): ViT 预训练分辨率，用于位置编码插值。
        weights_type (str): 权重类型，决定加载方式:
            - 'auto' (默认): 根据文件名自动判断（含 -XXXXXXXX.pth 哈希视为官方）。
            - 'official': 官方 DINOv3 权重，走 torch.hub.load 标准流程。
            - 'self_trained': 自研权重，按 student checkpoint 结构清理 key 后
              strict=False 加载（参考 load_checkpoint_forward.py）。
        untie_global_and_local_cls_norm (bool | None): 仅对自研权重生效,
            指定 ViT 架构的 untie_global_and_local_cls_norm。None 时自动从
            checkpoint 推断；自研权重若基于 SAT493M 通常需要 True。
        **kwargs: 透传给 DINOv3_Adapter 的其他参数。
    """

    def __init__(
        self,
        out_channels=128,
        dino_weight="/mnt/ht2-nas2/00-model/00-wj/Codes/checkpoints/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
        device="cuda",
        extract_ids=[5, 11, 17, 23],
        freeze_mode="frozen",
        unfreeze_last_n=0,
        pretrain_size=512,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=16,
        drop_path_rate=0.3,
        deform_ratio=0.5,
        with_cp=True,
        weights_type="auto",
        untie_global_and_local_cls_norm=None,
        **kwargs,
    ):
        super().__init__()

        # 解析 weights_type：'auto' 时按文件名自动判断
        if weights_type == "auto":
            is_self_trained = not _is_official_dino_weights(dino_weight)
        elif weights_type == "official":
            is_self_trained = False
        elif weights_type == "self_trained":
            is_self_trained = True
        else:
            raise ValueError(
                f"Unknown weights_type: '{weights_type}', expected one of "
                f"['auto', 'official', 'self_trained']"
            )

        if not is_self_trained:
            # 官方权重：走 torch.hub.load 的标准下载/加载流程
            backbone = torch.hub.load(
                REPO_DIR, DINO_NAME, source="local", weights=dino_weight
            )
        else:
            # 自研权重：先构建无预训练权重的 ViT，再按 student checkpoint
            # 的结构清理 key 后 strict=False 加载（参考 load_checkpoint_forward.py）
            logger.info(f"Loading self-trained DINOv3 weights from: {dino_weight}")
            # 只加载一次 checkpoint
            ckpt = torch.load(dino_weight, map_location="cpu", weights_only=False)
            raw_state = _extract_self_dino_state_dict(ckpt)
            cleaned = _clean_self_state_dict(raw_state)
            # 自动推断 untie_global_and_local_cls_norm
            untie = untie_global_and_local_cls_norm
            if untie is None:
                untie = any("cls_norm" in k for k in cleaned.keys())
                logger.info(
                    f"Auto-inferred untie_global_and_local_cls_norm={untie} "
                    f"from checkpoint keys"
                )
            # 直接调用 _make_dinov3_vit 构建 ViT-L（pretrained=False，避免下载），
            # 并显式传入 untie_global_and_local_cls_norm（dinov3_vitl16 不允许通过
            # kwargs 覆盖该参数）。架构参数与 dinov3_vitl16 保持一致。
            from dinov3.hub.backbones import _make_dinov3_vit
            backbone = _make_dinov3_vit(
                img_size=224,
                patch_size=16,
                in_chans=3,
                pos_embed_rope_base=100,
                pos_embed_rope_normalize_coords="separate",
                pos_embed_rope_rescale_coords=2,
                pos_embed_rope_dtype="fp32",
                embed_dim=1024,
                depth=24,
                num_heads=16,
                ffn_ratio=4,
                qkv_bias=True,
                drop_path_rate=0.0,
                layerscale_init=1.0e-05,
                norm_layer="layernormbf16",
                ffn_layer="mlp",
                ffn_bias=True,
                proj_bias=True,
                n_storage_tokens=4,
                mask_k_bias=True,
                untie_global_and_local_cls_norm=untie,
                pretrained=False,
                compact_arch_name="vitl",
            )
            load_self_dino_weights_from_state(backbone, cleaned)
        backbone = backbone.eval()
        if device is not None:
            backbone = backbone.to(device)

        init_requires_grad = freeze_mode in ('full_finetune', 'unfreeze_last_n')
        self.adapter = DINOv3_Adapter(
            backbone=backbone,
            interaction_indexes=list(extract_ids),
            pretrain_size=pretrain_size,
            conv_inplane=conv_inplane,
            n_points=n_points,
            deform_num_heads=deform_num_heads,
            drop_path_rate=drop_path_rate,
            deform_ratio=deform_ratio,
            with_cp=with_cp,
            backbone_requires_grad=init_requires_grad,
        )

        embed_dim = backbone.embed_dim
        # embed_dim(1024) -> out_channels 通道投影，复用 SepAdapterBlock
        self.proj = nn.ModuleList(
            [
                SepAdapterBlock(
                    in_dim=embed_dim, out_dim=out_channels, r=out_channels // 2
                )
                for _ in range(4)
            ]
        )

        # 应用初始冻结模式（处理 unfreeze_last_n 的部分解冻）
        self._current_freeze_mode = None
        self._current_unfreeze_n = 0
        self.set_freeze_mode(freeze_mode, unfreeze_last_n)

    def set_freeze_mode(self, mode='frozen', unfreeze_last_n=0):
        """动态切换 ViT 主干的冻结模式。

        可在训练过程中调用（如通过 FreezeScheduleHook），切换后 optimizer
        会自动开始更新新解冻的参数（因为所有参数在初始化时已加入 optimizer，
        冻结时 grad=None 被跳过，解冻后梯度正常计算）。

        Args:
            mode (str): 'frozen' / 'full_finetune' / 'unfreeze_last_n'
            unfreeze_last_n (int): mode='unfreeze_last_n' 时解冻的 block 数
        """
        vit = self.adapter.backbone

        if mode == 'frozen':
            vit.requires_grad_(False)
            self.adapter.backbone_requires_grad = False
        elif mode == 'full_finetune':
            vit.requires_grad_(True)
            self.adapter.backbone_requires_grad = True
        elif mode == 'unfreeze_last_n':
            vit.requires_grad_(False)
            n_blocks = len(vit.blocks)
            start = max(0, n_blocks - unfreeze_last_n)
            for i in range(start, n_blocks):
                vit.blocks[i].requires_grad_(True)
            # 解冻最终 norm 层
            if hasattr(vit, 'norm') and vit.norm is not None:
                for p in vit.norm.parameters():
                    p.requires_grad_(True)
            if hasattr(vit, 'cls_norm') and vit.cls_norm is not None:
                for p in vit.cls_norm.parameters():
                    p.requires_grad_(True)
            self.adapter.backbone_requires_grad = True
        else:
            raise ValueError(
                f"Unknown freeze_mode: '{mode}', expected one of "
                f"['frozen', 'full_finetune', 'unfreeze_last_n']")

        self._current_freeze_mode = mode
        self._current_unfreeze_n = unfreeze_last_n

    def forward(self, x):
        outs = self.adapter(x)  # dict {"1":..,"4":..}, 通道 = embed_dim
        feats = [outs["1"], outs["2"], outs["3"], outs["4"]]
        feats = [proj(f) for proj, f in zip(self.proj, feats)]
        return feats
