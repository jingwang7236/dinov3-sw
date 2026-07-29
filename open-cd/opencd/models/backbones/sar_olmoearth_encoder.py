# sar_olmoearth_encoder.py
# 基于 OlmoEarth 预训练模型的 SAR 图像编码器 (忠实复刻官方 FlexiHelios/Galileo encoder)
#
# 旧实现使用 nn.TransformerEncoder 重写, 键名与官方 checkpoint 完全不一致,
# 导致加载自研/开源 OlmoEarth 权重时几乎所有参数都 missing (静默随机初始化)。
#
# 本实现按官方 OlmoEarth encoder 的真实结构构建子模块, 使下列 checkpoint 键名
# 能被 1:1 完整加载 (开源 OlmoEarth-v1-Base 与自研 olmoearth10m_base 均兼容):
#   patch_embeddings.per_modality_embeddings.<mod>.<mod>__0.proj       (Conv2d)
#   composite_encodings.per_modality_channel_embeddings.<mod>          (Parameter)
#   composite_encodings.month_embed                                    (Embedding)
#   blocks.<i>.norm1 / attn.{q,k,v,proj} / norm2 / mlp.{fc1,fc2}
#   norm                                                               (LayerNorm)
#   project_and_aggregate.projection.0                                 (Linear)

import os
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from opencd.registry import MODELS


# ---------------------------------------------------------------------------
# 位置 / 时间编码 (与官方 olmoearth_pretrain.nn.encodings 完全一致, 此处内联避免引入重依赖)
# ---------------------------------------------------------------------------
BASE_GSD = 10  # 官方基准地面分辨率 (10m)


def get_1d_sincos_pos_encoding(pos, encoding_dim):
    assert encoding_dim % 2 == 0
    omega = torch.arange(encoding_dim // 2, device=pos.device) / encoding_dim / 2.0
    omega = 1.0 / (10000 ** omega)  # (D/2,)
    pos = pos.reshape(-1)           # (L,)
    out = torch.einsum('l,d->ld', pos, omega)  # (L, D/2)
    encoding = torch.cat([torch.sin(out), torch.cos(out)], dim=1)  # (L, D)
    return encoding


def get_2d_sincos_pos_encoding(grid, encoding_dim):
    assert encoding_dim % 2 == 0
    encoding_dim_1d = encoding_dim // 2
    emb_h = get_1d_sincos_pos_encoding(grid[0], encoding_dim_1d)
    emb_w = get_1d_sincos_pos_encoding(grid[1], encoding_dim_1d)
    return torch.cat([emb_h, emb_w], dim=1)


def get_2d_sincos_pos_encoding_with_resolution(grid_size, res, encoding_dim, device):
    """官方实现: 按地面分辨率缩放后的 2D sincos 空间位置编码。

    Args:
        grid_size: (H, W) patch 网格
        res: 每个 patch 代表的地面距离 (gsd_ratio = input_res * patch_size / BASE_GSD)
        encoding_dim: 编码维度 (embedding_size * 0.25)
    返回: (1, H*W, encoding_dim)
    """
    grid_h_size, grid_w_size = grid_size
    grid_h = torch.arange(grid_h_size, device=device)
    grid_w = torch.arange(grid_w_size, device=device)
    grid = torch.meshgrid(grid_w, grid_h, indexing='xy')
    grid = torch.stack(grid, dim=0)            # 2 x h x w
    grid = torch.einsum('chw,n->cnhw', grid, res)  # 2 x n x h x w
    _, n, h, w = grid.shape
    pos_embed = get_2d_sincos_pos_encoding(grid, encoding_dim)  # (n*H*W, D/2)? -> (n, H*W, D)
    pos_embed = pos_embed.reshape(n, h * w, encoding_dim)
    return pos_embed


def get_month_encoding_table(encoding_dim):
    assert encoding_dim % 2 == 0
    angles = torch.arange(0, 13) / (12 / (2 * np.pi))
    dim_per_table = encoding_dim // 2
    sin_table = torch.sin(torch.stack([angles for _ in range(dim_per_table)], axis=-1))
    cos_table = torch.cos(torch.stack([angles for _ in range(dim_per_table)], axis=-1))
    month_table = torch.concatenate([sin_table[:-1], cos_table[:-1]], axis=-1)
    return month_table  # (12, D)


# ---------------------------------------------------------------------------
# Transformer Block (与官方 nn.attention.Block / Attention 键名一致: 独立 q/k/v/proj)
# ---------------------------------------------------------------------------
class OlmoAttention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=True):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        x = F.scaled_dot_product_attention(q, k, v)  # (B, H, N, D)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class OlmoBlock(nn.Module):
    """官方 pre-norm Transformer block (无 LayerScale, 配置 init_values=None)."""

    def __init__(self, dim, num_heads=12, mlp_ratio=4.0, qkv_bias=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = OlmoAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential()  # 占位, 下面填充 fc1/fc2 以匹配键名
        self.mlp.fc1 = nn.Linear(dim, int(dim * mlp_ratio))
        self.mlp.fc2 = nn.Linear(int(dim * mlp_ratio), dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        h = self.act(self.mlp.fc1(self.norm2(x)))
        h = self.mlp.fc2(h)
        x = x + h
        return x


# ---------------------------------------------------------------------------
# 多尺度特征提取 (新增 adapter, 随机初始化 + 可训练; 不属于预训练 checkpoint)
# ---------------------------------------------------------------------------
class MultiScaleFeatureExtractor(nn.Module):
    """从 ViT 输出的 patch 网格特征中提取 4 个尺度特征 (conv 1x1 + avg_pool 下采样)."""

    def __init__(self, embed_dim, out_channels):
        super().__init__()
        self.proj2 = nn.Conv2d(embed_dim, out_channels, kernel_size=1)
        self.proj3 = nn.Conv2d(embed_dim, out_channels, kernel_size=1)
        self.proj4 = nn.Conv2d(embed_dim, out_channels, kernel_size=1)
        self.proj5 = nn.Conv2d(embed_dim, out_channels, kernel_size=1)

    def forward(self, x):
        # x: (B, N, C), 需先 reshape 回 patch 网格
        B, N, C = x.shape
        Hp = int(round(N ** 0.5))
        Wp = N // Hp
        x = x[:, :Hp * Wp].reshape(B, Hp, Wp, C).permute(0, 3, 1, 2).contiguous()
        feat2 = self.proj2(x)
        feat3 = self.proj3(F.avg_pool2d(x, kernel_size=2, stride=2))
        feat4 = self.proj4(F.avg_pool2d(x, kernel_size=4, stride=4))
        feat5 = self.proj5(F.avg_pool2d(x, kernel_size=8, stride=8))
        return [feat2, feat3, feat4, feat5]


@MODELS.register_module()
class OlmoEarthSAREncoder(nn.Module):
    """基于 OlmoEarth 预训练 encoder 的 SAR 编码器 (忠实复刻官方结构)。

    输入: SAR 图像 (B, C, H, W) — 支持任意 H, W (建议 32 的倍数)
    输出: 4 个尺度的特征列表 [p2, p3, p4, p5], 通道数 out_channels,
          空间尺寸对齐到原始输入的 [H/4, H/8, H/16, H/32] (与 OPT 分支一致)。

    权重加载: 严格按官方 encoder 键名加载, 若路径不存在或关键参数缺失则终止训练,
              杜绝静默随机初始化。两个 checkpoint 均兼容:
              - 开源 OlmoEarth-v1-Base (encoder.* 前缀)
              - 自研 olmoearth10m_base (含 encoder./target_encoder./decoder. 等前缀)
    """

    def __init__(self,
                 model_dir,
                 config_path,
                 weights_path,
                 model_variant='base',
                 patch_size=8,
                 image_size=256,
                 in_channels=3,
                 out_channels=128,
                 freeze_backbone=True,
                 adaptive_pool=False,
                 native_inference=True,
                 native_size=256,
                 modality='sentinel1',
                 input_res=10,
                 default_month=0,
                 load_projection=True,
                 strict_load=True):
        super().__init__()
        self.model_dir = model_dir
        self.config_path = config_path
        self.weights_path = weights_path
        self.modality = modality
        self.input_res = input_res
        self.default_month = int(default_month)
        self.freeze_backbone = freeze_backbone
        self.adaptive_pool = adaptive_pool
        self.native_inference = native_inference
        self.native_size = native_size
        self.out_channels = out_channels
        self.strict_load = strict_load

        if model_variant == 'base':
            embed_dim, depth, num_heads = 768, 12, 12
        elif model_variant == 'large':
            embed_dim, depth, num_heads = 1024, 24, 16
        else:
            raise ValueError(f"Unsupported model variant: {model_variant}")
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.in_channels = in_channels

        # --- 1. patch embedding (Conv2d, 与官方 FlexiPatchEmbed.proj 一致) ---
        # in_chans / patch 内核从 checkpoint 动态读取, 此处先以默认值占位, 在 load 时校正。
        self.pretrained_in_chans = None  # 由 _load_pretrained_weights 填充
        self.pretrained_patch = patch_size
        self.input_adapter = None  # 若输入通道 != 预训练通道, 用 1x1 conv 适配

        patch_embed = nn.ModuleDict()
        mod_dict = nn.ModuleDict()
        mod_dict[f'{modality}__0'] = _OlmoPatchProj(in_channels, embed_dim, patch_size)
        patch_embed[modality] = mod_dict
        self.patch_embeddings = nn.Module()
        self.patch_embeddings.per_modality_embeddings = patch_embed

        # --- 2. composite encodings ---
        # 每种编码占 embed_dim 的 1/4 (与官方一致)
        self.enc_dim_per_type = embed_dim // 4
        comp = nn.Module()
        comp.month_embed = nn.Embedding.from_pretrained(
            get_month_encoding_table(self.enc_dim_per_type), freeze=True)
        comp.per_modality_channel_embeddings = nn.ParameterDict()
        comp.per_modality_channel_embeddings[modality] = nn.Parameter(
            torch.zeros(1, self.enc_dim_per_type))
        self.composite_encodings = comp

        # --- 3. transformer blocks ---
        self.blocks = nn.ModuleList([
            OlmoBlock(embed_dim, num_heads=num_heads, mlp_ratio=4.0, qkv_bias=True)
            for _ in range(depth)])

        # --- 4. final norm ---
        self.norm = nn.LayerNorm(embed_dim)

        # --- 5. project_and_aggregate (预训练存在, 加载保持一致; 不参与 forward) ---
        self.project_and_aggregate = None
        if load_projection:
            proj_mod = nn.Module()
            proj_mod.projection = nn.Sequential(nn.Linear(embed_dim, embed_dim))
            self.project_and_aggregate = proj_mod

        # --- 6. 多尺度特征提取 (新增 adapter) ---
        self.feature_extractor = MultiScaleFeatureExtractor(embed_dim, out_channels)

        # --- 7. 自适应池化 (可选) ---
        if self.adaptive_pool:
            base = native_size // 4
            self.adapt_p2 = nn.AdaptiveAvgPool2d((base, base))
            self.adapt_p3 = nn.AdaptiveAvgPool2d((base // 2, base // 2))
            self.adapt_p4 = nn.AdaptiveAvgPool2d((base // 4, base // 4))
            self.adapt_p5 = nn.AdaptiveAvgPool2d((base // 8, base // 8))

        # 加载权重 (内部会校正 patch embed 的 in_chans / kernel 并按需创建 adapter)
        self._load_pretrained_weights()
        if self.freeze_backbone:
            self._freeze_backbone()

    # ------------------------------------------------------------------
    def _build_filtered_state_dict(self, raw_sd):
        """从原始 checkpoint 抽取属于本 encoder 的权重并去除前缀。

        优先使用 'encoder.' 前缀; 若不存在则视为裸 encoder 权重。
        仅保留与当前模型 state_dict 同名的键 (剔除 decoder./target_encoder. 等)。
        """
        model_keys = set(self.state_dict().keys())

        # 判断是否存在 'encoder.' 前缀
        has_enc_prefix = any(k.startswith('encoder.') for k in raw_sd.keys())
        if has_enc_prefix:
            prefix = 'encoder.'
            raw = {k[len(prefix):]: v for k, v in raw_sd.items()
                   if k.startswith(prefix)}
        else:
            raw = dict(raw_sd)

        # 官方 _drop_pos_embed_hook: pos_embed 是 legacy, 动态计算 sincos, 丢弃
        raw.pop('composite_encodings.pos_embed', None)

        filtered = {k: v for k, v in raw.items() if k in model_keys}
        skipped = [k for k in raw.keys() if k not in model_keys]
        return filtered, skipped

    def _load_pretrained_weights(self):
        """从本地 checkpoint 完整加载 encoder 权重。

        路径不存在或关键权重缺失时直接抛异常终止训练。
        """
        if not os.path.isfile(self.weights_path):
            raise FileNotFoundError(
                f"OlmoEarth weights file not found: {self.weights_path}. "
                f"训练终止: 请检查 OlmoEarth_weights_path 配置。")

        print(f"Loading OlmoEarth weights from {self.weights_path} ...")
        ckpt = torch.load(self.weights_path, map_location='cpu')
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']

        # --- 校正 patch embed 的 in_chans / kernel, 使其与 checkpoint 一致 ---
        pe_key_w = (f'patch_embeddings.per_modality_embeddings.{self.modality}'
                    f'.{self.modality}__0.proj.weight')
        if pe_key_w not in ckpt and ('encoder.' + pe_key_w) in ckpt:
            ckpt = {k[len('encoder.'):]: v for k, v in ckpt.items()
                    if k.startswith('encoder.')}
        if pe_key_w not in ckpt:
            raise KeyError(
                f"checkpoint 中找不到 patch embed 权重 '{pe_key_w}', "
                f"无法加载 modality='{self.modality}'。请确认配置的 modality 与 checkpoint 匹配。")

        conv_w = ckpt[pe_key_w]
        pretrained_out, pretrained_in, kh, kw = conv_w.shape
        self.pretrained_in_chans = pretrained_in
        self.pretrained_patch = kh
        if kh != kw:
            raise ValueError(f"OlmoEarth patch kernel 非方形: ({kh},{kw})")

        # 用 checkpoint 形状重建 patch embed conv, 保证严格加载
        proj = nn.Conv2d(pretrained_in, pretrained_out,
                         kernel_size=kh, stride=kh, bias=True)
        mod_dict = self.patch_embeddings.per_modality_embeddings[self.modality]
        mod_dict[f'{self.modality}__0'] = _OlmoPatchProj(
            pretrained_in, pretrained_out, kh, _share=proj)

        # 输入通道适配: BRIGHT SAR 为 3 通道, 预训练 conv 为 2 通道
        if self.in_channels != pretrained_in:
            print(f"  输入通道({self.in_channels}) != 预训练通道({pretrained_in}), "
                  f"创建 1x1 可训练 adapter: {self.in_channels}->{pretrained_in}")
            self.input_adapter = nn.Conv2d(self.in_channels, pretrained_in, 1, bias=True)

        # --- 抽取并加载属于本 encoder 的权重 ---
        filtered, skipped = self._build_filtered_state_dict(ckpt)
        missing, unexpected = self.load_state_dict(filtered, strict=False)

        # month_embed 是冻结的预训练表, 若加载后仍为默认(未命中)则提示
        # feature_extractor.* 与 input_adapter.* 是本编码器新增的多尺度投影头 / 通道适配器,
        # 不属于预训练 checkpoint, 理应随机初始化, 不计入"加载失败"。
        _new_prefixes = ('feature_extractor.', 'input_adapter.')
        me_missing = [k for k in missing if not k.startswith(_new_prefixes)]
        if me_missing:
            print(f"  ⚠️ Missing (pretrained) keys (first 10): {me_missing[:10]} ... "
                  f"(total {len(me_missing)})")
        new_missing = [k for k in missing if k.startswith(_new_prefixes)]
        if new_missing:
            print(f"  ℹ️ New params (random init, expected): "
                  f"{len(new_missing)} (feature_extractor & input_adapter)")
        if unexpected:
            print(f"  ⚠️ Unexpected keys (first 10): {unexpected[:10]} ... (total {len(unexpected)})")

        # 仅打印少量被跳过的 checkpoint 键 (decoder/target_encoder 等本就应跳过)
        if skipped:
            print(f"  ℹ️ Skipped non-encoder keys in checkpoint: {len(skipped)} "
                  f"(e.g. {skipped[:3]})")

        if self.strict_load and me_missing:
            raise RuntimeError(
                f"OlmoEarth 权重加载不完整, {len(me_missing)} 个预训练参数未加载 (missing)。"
                f"请检查 checkpoint 与配置是否匹配。Missing 示例: {me_missing[:5]}")

        loaded = len(filtered) - len(missing)
        n_pretrained = sum(1 for k in self.state_dict() if not k.startswith(_new_prefixes))
        print(f"✓ OlmoEarth pretrained weights loaded: {n_pretrained - len(me_missing)}/"
              f"{n_pretrained} (new adapter params kept random: {len(new_missing)})")

    def _freeze_backbone(self):
        """冻结主干网络参数 (patch embed / composite / blocks / norm / projection)。
        多尺度 feature_extractor 与 input_adapter 保持可训练。
        """
        for p in self.patch_embeddings.parameters():
            p.requires_grad = False
        for p in self.composite_encodings.parameters():
            p.requires_grad = False
        for p in self.blocks.parameters():
            p.requires_grad = False
        for p in self.norm.parameters():
            p.requires_grad = False
        if self.project_and_aggregate is not None:
            for p in self.project_and_aggregate.parameters():
                p.requires_grad = False
        print("✓ OlmoEarth backbone frozen (feature_extractor & input_adapter trainable).")

    # ------------------------------------------------------------------
    def _apply_composite_encodings(self, tokens, Hp, Wp):
        """与官方 CompositeEncodings._apply_encodings_per_modality 一致的加法编码。

        tokens: (B, Hp, Wp, num_bandsets=1, D) -> 返回同形状。
        4 段 (每段 D/4): [channel, time, month, space]
        """
        B = tokens.shape[0]
        D = tokens.shape[-1]
        n = self.enc_dim_per_type
        device = tokens.device
        embed = torch.zeros_like(tokens)

        # channel embedding (num_bandsets=1)
        ch = self.composite_encodings.per_modality_channel_embeddings[self.modality]
        embed[..., :n] = embed[..., :n] + ch[0].view(1, 1, 1, 1, n)

        # time encoding (单时相 t=1 -> position 0)
        time_enc = get_1d_sincos_pos_encoding(torch.zeros(1, device=device), n)  # (1, n)
        embed[..., n:2 * n] = embed[..., n:2 * n] + time_enc[0].view(1, 1, 1, 1, n)

        # month encoding
        month_idx = torch.tensor([self.default_month], device=device)
        month_emb = self.composite_encodings.month_embed(month_idx)  # (1, n)
        embed[..., 2 * n:3 * n] = embed[..., 2 * n:3 * n] + month_emb[0].view(1, 1, 1, 1, n)

        # spatial sincos (按地面分辨率缩放)
        gsd_ratio = self.input_res * self.pretrained_patch / BASE_GSD
        res = torch.ones(1, device=device) * gsd_ratio
        spatial = get_2d_sincos_pos_encoding_with_resolution(
            (Hp, Wp), res, n, device)  # (1, Hp*Wp, n)
        spatial = spatial.view(1, Hp, Wp, 1, n)
        embed[..., 3 * n:4 * n] = embed[..., 3 * n:4 * n] + spatial

        return tokens + embed

    def forward(self, x):
        """Args: x: (B, C, H, W) SAR 图像. Returns: [p2, p3, p4, p5]."""
        B, _, H_in, W_in = x.shape

        # 0. 内部下采样到 native_size 再过 ViT, 避免 token 数爆炸
        if self.native_inference and (H_in != self.native_size or W_in != self.native_size):
            x_in = F.interpolate(x, size=(self.native_size, self.native_size),
                                 mode='bilinear', align_corners=False)
        else:
            x_in = x

        # 1. 输入通道适配
        if self.input_adapter is not None:
            x_in = self.input_adapter(x_in)

        # 2. patch embedding (Conv2d) -> (B, D, Hp, Wp)
        proj = (self.patch_embeddings.per_modality_embeddings[self.modality]
                [f'{self.modality}__0'].proj)
        feat = proj(x_in)
        D, Hp, Wp = feat.shape[1], feat.shape[2], feat.shape[3]

        # (B, Hp, Wp, D) -> (B, Hp, Wp, 1, D) 以匹配 composite encodings 的 bandset 维
        tokens = feat.permute(0, 2, 3, 1).contiguous().unsqueeze(-2)

        # 3. composite encodings
        tokens = self._apply_composite_encodings(tokens, Hp, Wp)

        # flatten: (B, Hp*Wp, D)
        tokens = tokens.reshape(B, Hp * Wp, D)

        # 4. transformer blocks
        for blk in self.blocks:
            tokens = blk(tokens)

        # 5. final norm
        tokens = self.norm(tokens)

        # 6. 多尺度特征
        feats = self.feature_extractor(tokens)  # [p2,p3,p4,p5] @ (B, out, Hp, Wp)/...

        # 7. 自适应池化对齐 (可选)
        if self.adaptive_pool:
            feats[0] = self.adapt_p2(feats[0])
            feats[1] = self.adapt_p3(feats[1])
            feats[2] = self.adapt_p4(feats[2])
            feats[3] = self.adapt_p5(feats[3])

        # 8. 对齐到标准多尺度尺寸 [H/4, H/8, H/16, H/32], 与光学分支 (DINOv3
        #    Adapter, strides [4,8,16,32]) 保持一致。feature_extractor 实际输出
        #    按 token 网格池化 (patch_size=8 时为 strides [8,16,32,64]), 即便
        #    输入等于 native_size 也必须 resize, 否则双分支尺度不匹配会导致
        #    CrossAttnFusion 等融合模块在窗口切分时崩溃。
        if self.native_inference:
            strides = [4, 8, 16, 32]
            for i, s in enumerate(strides):
                tgt = (H_in // s, W_in // s)
                if feats[i].shape[-2:] != tgt:
                    feats[i] = F.interpolate(
                        feats[i], size=tgt, mode='bilinear', align_corners=False)

        return feats


class _OlmoPatchProj(nn.Module):
    """包裹 Conv2d 的 patch 投影, 键名为 `.proj.*`, 与官方 FlexiPatchEmbed 对齐。"""

    def __init__(self, in_chans, embed_dim, patch_size, _share=None):
        super().__init__()
        self.proj = _share if _share is not None else nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)
