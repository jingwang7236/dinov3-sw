"""蒸馏 Swin Transformer backbone — 完全自包含，无外部依赖。

将蒸馏得到的 Swin-huge 模型作为特征提取器，直接提取 4 级层级特征
[p2, p3, p4, p5]（对应 /4, /8, /16, /32），经 1x1 卷积投影到统一通道数，
可直接被 ChangeDinoDecoder / ChangeDinoCrossAttnDecoder 消费。

Swin Transformer 实现已内联到本文件中，不依赖外部 dinov3_distill 代码库，
方便项目迁移到其他机器。

支持三种训练模式，可通过 FreezeScheduleHook 在指定迭代次数切换:
  - frozen:          完全冻结 Swin 主干，仅训练投影层
  - unfreeze_last_n: 解冻最后 N 个 transformer block + norm 层
  - full_finetune:   全量微调
"""
import logging
import math
from contextlib import nullcontext
from functools import partial
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from opencd.registry import MODELS

logger = logging.getLogger(__name__)


# ===========================================================================
# 工具函数 (替代 timm 依赖)
# ===========================================================================

def to_2tuple(x):
    if isinstance(x, (tuple, list)):
        return x
    return (x, x)


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """截断正态分布初始化 (与 torch.nn.init.trunc_normal_ 一致)。"""
    nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample。"""

    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


# ===========================================================================
# Swin Transformer 组件
# (copy-paste from Swin-Transformer, 精简为仅保留下游特征提取所需部分)
# ===========================================================================

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """(B, H, W, C) -> (num_windows*B, window_size, window_size, C)"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size,
               W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(
        -1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """(num_windows*B, window_size, window_size, C) -> (B, H, W, C)"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) with relative position bias."""

    def __init__(self, dim, window_size, num_heads, qkv_bias=True,
                 qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def reset_relative_position_index(self):
        """在当前设备上重新计算 relative_position_index buffer。"""
        device = self.relative_position_index.device
        dtype = self.relative_position_index.dtype
        coords_h = torch.arange(self.window_size[0], device=device, dtype=dtype)
        coords_w = torch.arange(self.window_size[1], device=device, dtype=dtype)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.relative_position_index.copy_(relative_position_index)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_, self.num_heads, N, N) + mask.unsqueeze(1)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn_out = attn
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn_out


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block (W-MSA / SW-MSA)。"""

    def __init__(self, dim, input_resolution, num_heads, window_size=7,
                 shift_size=0, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        self.H = input_resolution[0]
        self.W = input_resolution[1]
        self.attn_mask_dict = {}

    def create_attn_mask(self, H, W):
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1))
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)) \
                          .masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def _forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = H

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size),
                                   dims=(1, 2))
            if H not in self.attn_mask_dict:
                self.attn_mask_dict[H] = self.create_attn_mask(H, W) \
                    .to(x.device).to(x.dtype)
            attn_mask = self.attn_mask_dict[H]
            attn_mask = attn_mask.unsqueeze(0).repeat(B, 1, 1, 1)
            attn_mask = attn_mask.reshape(
                -1, self.window_size * self.window_size,
                self.window_size * self.window_size)
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows, attn = self.attn(x_windows, attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size,
                                          self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size),
                           dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn

    def forward(self, x):
        return self._forward(x)


class PatchMerging(nn.Module):
    """Patch Merging Layer (2x 下采样)。"""

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = H
        x = x.view(B, H, W, C)

        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class BasicLayer(nn.Module):
    """Swin Transformer 的一个 stage (含可选 PatchMerging)。"""

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                 downsample=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x, _ = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding。"""

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96,
                 norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0],
                              img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x.transpose(1, 2).reshape(B, C, H, W)


class SwinTransformer(nn.Module):
    """Swin Transformer (仅保留下游特征提取所需部分)。

    去除了蒸馏特有的 feature_proj / proj_cls / head / masked_embed 等,
    这些 key 在 load_state_dict(strict=False) 时会被安全跳过。
    """

    def __init__(self, img_size=256, patch_size=4, in_chans=3,
                 embed_dim=96, depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24), window_size=7, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.2,
                 norm_layer=partial(nn.LayerNorm, eps=1e-5),
                 ape=False, patch_norm=True, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_layers = len(depths)
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = np.linspace(0, drop_path_rate, sum(depths)).tolist()
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                  patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)


# ===========================================================================
# 模型工厂
# ===========================================================================

def swin_tiny(window_size=8, **kwargs):
    return SwinTransformer(
        window_size=window_size, embed_dim=96, depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24], mlp_ratio=4, qkv_bias=True,
        drop_path_rate=kwargs.pop('drop_path_rate', 0.1), **kwargs)


def swin_small(window_size=8, **kwargs):
    return SwinTransformer(
        window_size=window_size, embed_dim=96, depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24], mlp_ratio=4,
        drop_path_rate=kwargs.pop('drop_path_rate', 0.2), **kwargs)


def swin_base(window_size=8, **kwargs):
    return SwinTransformer(
        window_size=window_size, embed_dim=128, depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32], mlp_ratio=4,
        drop_path_rate=kwargs.pop('drop_path_rate', 0.2),
        qkv_bias=kwargs.pop('qkv_bias', True), **kwargs)


def swin_large(window_size=8, **kwargs):
    return SwinTransformer(
        window_size=window_size, embed_dim=192, depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48], mlp_ratio=4,
        drop_path_rate=kwargs.pop('drop_path_rate', 0.2),
        qkv_bias=kwargs.pop('qkv_bias', True), **kwargs)


def swin_huge(window_size=8, **kwargs):
    return SwinTransformer(
        window_size=window_size, embed_dim=352, depths=[2, 2, 18, 2],
        num_heads=[8, 16, 32, 64], mlp_ratio=4,
        drop_path_rate=kwargs.pop('drop_path_rate', 0.2),
        qkv_bias=kwargs.pop('qkv_bias', True), **kwargs)


_SWIN_MODELS = {
    'swin_tiny': swin_tiny,
    'swin_small': swin_small,
    'swin_base': swin_base,
    'swin_large': swin_large,
    'swin_huge': swin_huge,
}


# ===========================================================================
# Backbone
# ===========================================================================

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

    直接提取 Swin 的 4 级层级特征，经 1x1 卷积投影到 out_channels。
    支持冻结 / 部分微调 / 全量微调三种训练模式。

    Args:
        out_channels (int): 输出通道数，应与 decode_head 的 fpn_channels 一致。
        swin_weight (str): 蒸馏 Swin 权重路径 (.pt)。
        device (str | None): 加载 Swin 时使用的设备，None 时不迁移。
        swin_model (str): Swin 模型规模，默认 'swin_huge'。
        freeze_mode (str): 初始冻结模式:
            - 'frozen': 完全冻结 Swin 主干，仅训练投影层 (默认)。
            - 'full_finetune': 全量微调，Swin 所有参数可训练。
            - 'unfreeze_last_n': 仅解冻最后 ``unfreeze_last_n`` 个 transformer
              block 及最终 norm 层。
        unfreeze_last_n (int): ``freeze_mode='unfreeze_last_n'`` 时解冻的 block 数。
        norm_cfg (dict): 投影层 norm 配置，默认 SyncBN。
        act_cfg (dict): 投影层激活函数配置，默认 SiLU。
        **kwargs: 透传给 swin 模型工厂的其他参数。

    Note:
        Swin-huge 架构: embed_dim=352, depths=[2,2,18,2],
        num_heads=[8,16,32,64], window_size=8, patch_size=4。
        4 级特征通道: [352, 704, 1408, 2816]。
    """

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
        **kwargs,
    ):
        super().__init__()

        if norm_cfg is None:
            norm_cfg = dict(type='SyncBN', requires_grad=True)
        if act_cfg is None:
            act_cfg = dict(type='SiLU', inplace=True)

        if swin_model not in _SWIN_MODELS:
            raise ValueError(
                f'Unknown swin_model: {swin_model!r}, '
                f'expected one of {list(_SWIN_MODELS.keys())}'
            )

        # 构建 Swin 模型
        # img_size=256 保证所有 stage 分辨率 >= window_size(8)，
        # 避免 SwinTransformerBlock 自动缩小 window_size 导致与 checkpoint 不匹配
        self.swin = _SWIN_MODELS[swin_model](
            img_size=256,
            patch_size=4,
            in_chans=3,
            ape=False,
            patch_norm=True,
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

        # 重置 relative_position_index buffer 到正确设备
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

        # 收集所有 block (用于 unfreeze_last_n)
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

        norm = nn.SyncBatchNorm if norm_type == 'SyncBN' else nn.BatchNorm2d

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

        Args:
            mode: 'frozen' / 'full_finetune' / 'unfreeze_last_n'
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
            for attr in ('norm',):
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
    # forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """前向传播，返回 4 级多尺度特征列表。

        Returns:
            [p2, p3, p4, p5]: 通道数为 out_channels，
            分别对应 H/4, H/8, H/16, H/32
        """
        ctx = torch.no_grad() if self._swin_frozen else nullcontext()

        with ctx:
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

                B2, L, C = x_tok.shape
                feat_2d = x_tok.transpose(1, 2).reshape(
                    B2, C, cur_h, cur_w).contiguous()
                raw_feats.append(feat_2d)

                if layer.downsample is not None:
                    x_tok = layer.downsample(x_tok)
                    cur_h, cur_w = cur_h // 2, cur_w // 2

        out = [proj(f) for proj, f in zip(self.proj, raw_feats)]
        return out


# ===========================================================================
# Adapter 模块 (轻量可训练, 用于 SwinDistillAdapterBackbone)
# ===========================================================================

class SwinFeatureAdapter(nn.Module):
    """单尺度特征 Adapter: 1x1 reduce → 3x3 dw → 1x1 proj。

    比 SwinDistillBackbone 的简单 1x1 Conv 投影多了 depthwise 空间建模能力，
    适合 Swin 冻结时以少量参数适配下游任务。

    Args:
        in_dim (int): 输入通道 (各 stage 的通道数)。
        out_dim (int): 输出通道 (应与 decode_head fpn_channels 一致)。
        r (int): bottleneck 维度, 默认等于 out_dim。
        norm_layer: 归一化层, 默认 SyncBatchNorm。
        act_layer: 激活函数, 默认 SiLU。
    """

    def __init__(self, in_dim, out_dim, r=None,
                 norm_layer=nn.SyncBatchNorm, act_layer=nn.SiLU):
        super().__init__()
        r = r or out_dim
        self.reduce = nn.Sequential(
            nn.Conv2d(in_dim, r, kernel_size=1, bias=False),
            norm_layer(r),
            act_layer(inplace=True),
        )
        self.dw = nn.Sequential(
            nn.Conv2d(r, r, kernel_size=3, padding=1, groups=r, bias=False),
            norm_layer(r),
            act_layer(inplace=True),
        )
        self.proj = nn.Conv2d(r, out_dim, kernel_size=1, bias=True)

    def forward(self, x):
        return self.proj(self.dw(self.reduce(x)))


# ===========================================================================
# SwinDistillAdapterBackbone — Swin 始终冻结, 仅训练 adapter
# ===========================================================================

@MODELS.register_module()
class SwinDistillAdapterBackbone(SwinDistillBackbone):
    """蒸馏 Swin Transformer + Adapter backbone。

    与 :class:`SwinDistillBackbone` 的区别:
      - SwinDistillBackbone: 简单 1x1 Conv 投影, 支持 frozen / full_finetune
      - SwinDistillAdapterBackbone: 使用 :class:`SwinFeatureAdapter`
        (1x1 reduce → 3x3 dw → 1x1 proj) 替代简单投影,
        Swin 主干始终冻结, 仅训练 adapter + decode_head。

    训练模式:
      - 默认: Swin 冻结, adapter 可训练 (freeze_mode='frozen')
      - 可选: 通过 FreezeScheduleHook 在指定迭代解冻 Swin (full_finetune /
        unfreeze_last_n), 与父类行为一致

    Args:
        adapter_r (int | None): adapter bottleneck 维度, None 时等于 out_channels。
        其余参数同 :class:`SwinDistillBackbone`。
    """

    def __init__(
        self,
        out_channels: int = 128,
        adapter_r: int = None,
        **kwargs,
    ):
        self._adapter_r = adapter_r or out_channels
        super().__init__(out_channels=out_channels, **kwargs)

    def _build_proj(self, in_dim, out_dim):
        """构建 adapter 替代简单投影层。"""
        return SwinFeatureAdapter(
            in_dim=in_dim,
            out_dim=out_dim,
            r=self._adapter_r,
            norm_layer=nn.SyncBatchNorm
            if self.norm_cfg.get('type', 'SyncBN') == 'SyncBN'
            else nn.BatchNorm2d,
            act_layer=nn.SiLU
            if self.act_cfg.get('type', 'SiLU') == 'SiLU'
            else nn.ReLU,
        )
