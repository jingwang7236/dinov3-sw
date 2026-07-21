# sar_olmoearth_encoder.py
# 基于 OlmoEarth 预训练模型的 SAR 图像编码器
# 特性：支持任意输入尺寸，自适应池化对齐特征图，无 transformers 依赖

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from opencd.registry import MODELS

class OlmoEarthPatchEmbedding(nn.Module):
    """
    支持任意输入尺寸的 Patch Embedding
    """
    def __init__(self, patch_size=4, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) - 支持任意 H, W
        B, C, H, W = x.shape
        x = self.proj(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        
        # 记录空间维度
        Hp, Wp = x.shape[2], x.shape[3]
        
        # Flatten 并转置
        x = x.flatten(2).transpose(1, 2)  # (B, Hp*Wp, embed_dim)
        
        return x, (Hp, Wp)  # 返回特征和空间维度

class MultiScaleFeatureExtractor(nn.Module):
    """
    从 Transformer 输出中提取多尺度特征
    支持动态输入尺寸，输出固定的 4 个尺度特征
    """
    def __init__(self, embed_dim, out_channels, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        
        # 投影层 - 全部使用 Conv2d 以支持空间维度
        self.proj2 = nn.Conv2d(embed_dim, out_channels, kernel_size=1)
        self.proj3 = nn.Conv2d(embed_dim, out_channels, kernel_size=1)
        self.proj4 = nn.Conv2d(embed_dim, out_channels, kernel_size=1)
        self.proj5 = nn.Conv2d(embed_dim, out_channels, kernel_size=1)
    
    def forward(self, x, Hp, Wp):
        """
        Args:
            x: Transformer 输出，形状 (B, N, embed_dim)，N = Hp * Wp
            Hp, Wp: Patch 网格的空间维度
        """
        B, N, C = x.shape
        
        # 将序列重塑回空间维度 (B, C, Hp, Wp)
        x = x.reshape(B, Hp, Wp, C).permute(0, 3, 1, 2)
        
        # 提取不同尺度的特征
        # /4 尺度 (对应原始 patch 网格分辨率)
        feat2 = self.proj2(x)  # (B, out_channels, Hp, Wp)
        
        # /8 尺度 (通过 stride=2 的池化下采样)
        feat3 = self.proj3(nn.functional.avg_pool2d(x, kernel_size=2, stride=2))
        
        # /16 尺度 (通过 stride=4 的池化下采样)
        feat4 = self.proj4(nn.functional.avg_pool2d(x, kernel_size=4, stride=4))
        
        # /32 尺度 (通过 stride=8 的池化下采样)
        feat5 = self.proj5(nn.functional.avg_pool2d(x, kernel_size=8, stride=8))
        
        return [feat2, feat3, feat4, feat5]

@MODELS.register_module()
class OlmoEarthSAREncoder(nn.Module):
    """
    基于 OlmoEarth 预训练模型的 SAR 图像编码器
    
    输入: SAR 图像 (B, 3, H, W) - 支持任意 H, W
    输出: 4 个尺度的特征列表 [p2, p3, p4, p5]
          经过自适应池化，形状固定为 (B, 128, 64, 64), (B, 128, 32, 32), (B, 128, 16, 16), (B, 128, 8, 8)
    """
    def __init__(self, 
                 model_dir,          # 模型目录
                 config_path,        # config.json 路径
                 weights_path,       # weights.pth 路径
                 model_variant='base',
                 patch_size=4,
                 image_size=256,
                 in_channels=3,
                 out_channels=128,
                 freeze_backbone=True,
                 adaptive_pool=False,
                 native_inference=True,
                 native_size=256):
        super().__init__()
        self.model_dir = model_dir
        self.config_path = config_path
        self.weights_path = weights_path
        self.model_variant = model_variant
        self.patch_size = patch_size
        self.image_size = image_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.freeze_backbone = freeze_backbone
        self.adaptive_pool = adaptive_pool
        # native_inference: 将输入在内部下采样到 native_size (256) 再过 ViT,
        # 避免大分辨率下 token 数爆炸导致自注意力算力/显存暴涨 (N² 复杂度)。
        # 例如 512 输入若直接过 patch_size=4 的 ViT 会产生 128×128=16384 tokens,
        # 注意力代价是 256(4096 tokens) 的 16 倍, 单 iter 极慢甚至接近 OOM。
        self.native_inference = native_inference
        self.native_size = native_size
        
        # 根据模型变体设置参数
        if model_variant == 'base':
            self.embed_dim = 768
            self.depth = 12
            self.num_heads = 12
        elif model_variant == 'large':
            self.embed_dim = 1024
            self.depth = 24
            self.num_heads = 16
        else:
            raise ValueError(f"Unsupported model variant: {model_variant}")
        
        # 1. Patch Embedding
        self.patch_embed = OlmoEarthPatchEmbedding(
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=self.embed_dim
        )
        
        # 2. 可学习的位置编码（初始化为 256x256 图像对应的 64x64 patches）
        # 最大支持 64*64 = 4096 个 patches
        self.pos_embed = nn.Parameter(torch.zeros(1, 64*64, self.embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # 3. Transformer Encoder (使用原生 PyTorch 实现)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.depth)
        
        # 4. 多尺度特征提取器
        self.feature_extractor = MultiScaleFeatureExtractor(
            embed_dim=self.embed_dim,
            out_channels=out_channels,
            patch_size=patch_size
        )
        
        # 5. 自适应池化层 (确保输出尺寸固定，解决 Cross-Attention 维度不匹配问题)
        if self.adaptive_pool:
            self.adapt_p2 = nn.AdaptiveAvgPool2d((64, 64))
            self.adapt_p3 = nn.AdaptiveAvgPool2d((32, 32))
            self.adapt_p4 = nn.AdaptiveAvgPool2d((16, 16))
            self.adapt_p5 = nn.AdaptiveAvgPool2d((8, 8))
        
        # 6. 初始化并加载权重
        self._load_pretrained_weights()
        if self.freeze_backbone:
            self._freeze_backbone()

    def _load_pretrained_weights(self):
        """从本地文件加载预训练权重"""
        try:
            # 读取配置
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            # 加载权重
            print(f"Loading weights from {self.weights_path}...")
            state_dict = torch.load(self.weights_path, map_location='cpu')
            
            # --- 权重键名映射逻辑 ---
            # 注意：OlmoEarth 的权重键名可能与此模型定义不一致。
            # 此处提供了一个通用的加载框架，如果不匹配，需要根据实际的 state_dict 打印结果进行映射。
            
            model_state_dict = self.state_dict()
            new_state_dict = {}
            
            # 简单的严格加载尝试
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"⚠️ Missing keys: {missing_keys[:5]}... (total {len(missing_keys)})")
            if unexpected_keys:
                print(f"⚠️ Unexpected keys (ignored): {unexpected_keys[:5]}... (total {len(unexpected_keys)})")
                
            print("✓ Weight loading process completed (check warnings above for issues).")
            
        except Exception as e:
            print(f"✗ Error loading pretrained weights: {e}")
            print("  Model will proceed with randomly initialized weights.")

    def _freeze_backbone(self):
        """冻结主干网络参数"""
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        for param in self.transformer.parameters():
            param.requires_grad = False
        if self.pos_embed is not None:
            self.pos_embed.requires_grad = False
        print("✓ OlmoEarth backbone frozen.")

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) 输入 SAR 图像
        Returns:
            list: [p2, p3, p4, p5]，多尺度特征，通道数为 out_channels。
            若 native_inference=True，特征空间尺寸对齐到原始输入的
            [H/4, H/8, H/16, H/32]，与 OPT 分支一致，可直接送入 decoder。
        """
        B, _, H_in, W_in = x.shape

        # 0. 内部下采样到 native_size 再过 ViT，避免 token 数爆炸
        if self.native_inference and (H_in != self.native_size or W_in != self.native_size):
            x_in = F.interpolate(
                x, size=(self.native_size, self.native_size),
                mode='bilinear', align_corners=False)
        else:
            x_in = x

        B = x_in.shape[0]
        
        # 1. Patch Embedding
        x, (Hp, Wp) = self.patch_embed(x_in)  # (B, N, embed_dim)
        N = Hp * Wp
        
        # 2. 位置编码处理 (支持动态尺寸插值)
        if self.pos_embed is not None:
            # 当前输入的 patch 数量
            current_n = N
            # 预训练的 patch 数量 (假设是 64*64)
            pretrained_n = self.pos_embed.shape[1]
            
            if current_n == pretrained_n:
                pos_embed = self.pos_embed
            elif current_n < pretrained_n:
                # 如果输入图像小，截取位置编码
                pos_embed = self.pos_embed[:, :current_n, :]
            else:
                # 如果输入图像大，使用双三次插值扩展位置编码
                # 1. Reshape 为 2D (1, C, H_grid, W_grid)
                pos_embed = self.pos_embed.reshape(1, 64, 64, self.embed_dim).permute(0, 3, 1, 2)
                # 2. Interpolate 到
                pos_embed = nn.functional.interpolate(
                    pos_embed, 
                    size=(Hp, Wp), 
                    mode='bicubic',
                    align_corners=False
                )
                # 3. Reshape 回 1D (1, N, C)
                pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, current_n, self.embed_dim)
            
            x = x + pos_embed

        # 3. Transformer 编码
        x = self.transformer(x)  # (B, N, embed_dim)
        
        # 4. 提取多尺度特征
        feats = self.feature_extractor(x, Hp, Wp)  # [p2, p3, p4, p5]
        
        # 5. 自适应池化对齐 (关键步骤：防止 Cross-Attention 报错)
        if self.adaptive_pool:
            feats[0] = self.adapt_p2(feats[0])  # 强制 -> (B, 128, 64, 64)
            feats[1] = self.adapt_p3(feats[1])  # 强制 -> (B, 128, 32, 32)
            feats[2] = self.adapt_p4(feats[2])  # 强制 -> (B, 128, 16, 16)
            feats[3] = self.adapt_p5(feats[3])  # 强制 -> (B, 128, 8, 8)

        # 6. 若启用了 native_inference 且发生过下采样，需把特征上采样回
        # 原始输入对应的多尺度尺寸 [H/4, H/8, H/16, H/32]，与 OPT 分支对齐。
        if self.native_inference and (H_in != self.native_size or W_in != self.native_size):
            strides = [4, 8, 16, 32]
            for i, s in enumerate(strides):
                tgt = (H_in // s, W_in // s)
                if feats[i].shape[-2:] != tgt:
                    feats[i] = F.interpolate(
                        feats[i], size=tgt, mode='bilinear', align_corners=False)

        return feats
