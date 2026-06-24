import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import re
import os
import sys
import contextlib
sys.path.insert(0, "/mnt/ht2-nas2/00-model/00-wj/Codes/dinov3-sw/open-cd")


# REPO_DIR = "dinov3"
REPO_DIR = "/mnt/ht2-nas2/00-model/00-wj/Codes/dinov3-sw/open-cd/dinov3"
DINO_NAME = "dinov3_vitl16"
MODEL_TO_NUM_LAYERS = {
    "VITS": 12,
    "VITSP": 12,
    "VITB": 12,
    "VITL": 24,
    "VITHP": 32,
    "VIT7B": 40,
}

class DINOV3Wrapper(nn.Module):
    """
    DINOv3 特征提取器包装类，支持灵活的权重训练策略。
    
    Args:
        weights_path (str): 预训练权重路径
        extract_ids (List[int]): 要提取的中间层索引
        device (str): 设备类型
        freeze_mode (str): 权重训练模式，可选:
            - 'frozen': 完全冻结（默认）
            - 'unfreeze_last_n': 解冻最后 N 层
            - 'full_finetune': 全量微调
        unfreeze_layers (int): 当 freeze_mode='unfreeze_last_n' 时，解冻的层数
        verbose (bool): 是否打印详细信息
    """
    def __init__(
        self,
        weights_path="/mnt/ht2-nas2/00-model/00-wj/Codes/checkpoints/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
        extract_ids=[5, 11, 17, 23],
        device="cuda",
        freeze_mode: str = "frozen",  # 'frozen', 'unfreeze_last_n', 'full_finetune'
        unfreeze_layers: int = 2,
    ):
        super().__init__()
        self.device = device
        self.freeze_mode = freeze_mode
        self.unfreeze_layers = unfreeze_layers
        self.model = torch.hub.load(
            REPO_DIR,
            DINO_NAME,
            source="local",
            weights=weights_path,
        )
        
        self.model = self.model.eval().to(device)
        self.n_layers = MODEL_TO_NUM_LAYERS[
            re.sub(r"\d+", "", DINO_NAME.split("_")[-1]).upper()
        ]
        self.patch_size = int(re.findall(r"\d+", DINO_NAME.split("_")[-1])[-1])
        self.extract_ids = extract_ids

        # freeze the backbone
        # for p in self.model.parameters():
        #     p.requires_grad = False

        self._apply_freeze_strategy()

    def _apply_freeze_strategy(self):
        """根据 freeze_mode 应用不同的权重训练策略"""

        if self.freeze_mode == "frozen":
            # 完全冻结
            self._freeze_all()

        elif self.freeze_mode == "full_finetune":
            # 全量微调
            self._unfreeze_all()

        elif self.freeze_mode == "unfreeze_last_n":
            # 解冻最后 N 层
            self._freeze_all()
            self._unfreeze_last_n_layers(self.unfreeze_layers)

        else:
            raise ValueError(
                f"Unsupported freeze_mode: {self.freeze_mode}. "
                f"Supported modes: 'frozen', 'unfreeze_last_n', 'full_finetune'"
            )
    def _freeze_all(self):
        """冻结所有参数"""
        for param in self.model.parameters():
            param.requires_grad = False
        print("🔒 DINOv3: All layers frozen")
    
    def _unfreeze_all(self):
        """解冻所有参数"""
        for param in self.model.parameters():
            param.requires_grad = True
        print("🔓 DINOv3: All layers unfrozen (full finetune)")
    
    def _unfreeze_last_n_layers(self, n: int):
        """解冻最后 N 层 Transformer blocks"""
        total_blocks = len(self.model.blocks)
        start_idx = max(0, total_blocks - n)
        
        # 解冻指定的 blocks
        for i in range(start_idx, total_blocks):
            for param in self.model.blocks[i].parameters():
                param.requires_grad = True
        
        # 同时解冻最后的 LayerNorm
        if hasattr(self.model, 'norm'):
            for param in self.model.norm.parameters():
                param.requires_grad = True
        
        print(f"🔓 DINOv3: Unfrozen last {n} layers (blocks {start_idx}-{total_blocks-1})")
    
    def forward(self, x):
        scale_factor = 2 / (512 / x.shape[-1])
        x = F.interpolate(
            x, size=(512, 512), mode="bilinear", align_corners=True, antialias=True
        )
        # 关键：冻结时用 no_grad 省显存；微调时必须放开梯度，否则
        # 即使 requires_grad=True，梯度也会在此被截断，DINOv3 无法被更新。
        if self.freeze_mode == "frozen":
            grad_ctx = torch.no_grad()
        else:
            grad_ctx = contextlib.nullcontext()

        with grad_ctx:
            feats = self.model.get_intermediate_layers(
                x, n=range(self.n_layers), reshape=True, norm=True
            )
            feats_ = []
            for i in range(len(self.extract_ids)):
                feats_.append(
                    F.interpolate(
                        feats[self.extract_ids[i]],
                        scale_factor=scale_factor,
                        mode="bilinear",
                    )
                )
        return feats_

class SepAdapterBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, r: int = 64, act=nn.SiLU):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_dim, r, kernel_size=1, bias=False),
            nn.BatchNorm2d(r),
            act(inplace=True),
        )
        self.dw = nn.Sequential(
            nn.Conv2d(
                r, r, kernel_size=3, padding=1, groups=r, bias=False
            ),  # depthwise
            nn.BatchNorm2d(r),
            act(inplace=True),
        )
        self.proj = nn.Conv2d(r, out_dim, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.reduce(x)
        x = self.dw(x)
        x = self.proj(x)
        return x


class DenseAdapterLite(nn.Module):
    def __init__(
        self,
        in_dim=1024,
        out_dim=256,
        bottleneck=64,
        share=False,
    ):
        super().__init__()
        if share:
            self.blocks = nn.ModuleList(
                [SepAdapterBlock(in_dim, out_dim, r=bottleneck)]
            )
        else:
            self.blocks = nn.ModuleList(
                [SepAdapterBlock(in_dim, out_dim, r=bottleneck) for _ in range(4)]
            )
        self.share = share

    def forward(self, feats):
        """
        feats: list of 4 tensors, each [B, C, H_i, W_i]（C = in_dim）
        return: list of 4 tensors, each [B, out_dim, S_i, S_i], S_i ∈ self.sizes
        """
        outs = []
        for i, x in enumerate(feats):
            x = F.interpolate(
                x,
                scale_factor=2 / (2**i),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            block = self.blocks[0] if self.share else self.blocks[i]
            outs.append(block(x))
        return outs


# ================================================================
# LoRA Plugin — 独立模块，不修改上方任何已有类
# 用法: apply_lora(vit_model, r=8, alpha=16)
# ================================================================
import math as _math


class LoRALinear(nn.Module):
    """LoRA 低秩适配器，包裹已有 nn.Linear。

    前向: y = W_base(x) + (alpha/r) * B(A(x))
    初始时 B=0 → 输出与原模型完全一致，训练中逐步学习增量。

    Args:
        base_linear: 原始 Linear 层（注入后自动冻结）
        r: LoRA 秩
        alpha: 缩放系数
        dropout: LoRA 路径 dropout
    """

    def __init__(self, base_linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.base = base_linear
        self.scale = alpha / r

        in_f, out_f = base_linear.in_features, base_linear.out_features
        self.lora_A = nn.Linear(in_f, r, bias=False)
        self.lora_B = nn.Linear(r, out_f, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=_math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.lora_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        for p in self.base.parameters():
            p.requires_grad = False

    # ---- 属性代理：让 LoRALinear 透明暴露原始 Linear 的常用属性 ----
    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(self.lora_drop(x))) * self.scale


def apply_lora(model, r=8, alpha=16, target_modules=None,
               dropout=0.0, verbose=True):
    """向 ViT 模型的每个 Transformer Block 注入 LoRA 适配器。

    注入后所有原始参数冻结，仅 lora_A / lora_B 可训练。

    Args:
        model: ViT 模型（需有 .blocks 属性，每个 block 含 .attn 和 .mlp）
        r: LoRA 秩 (推荐 4/8/16)
        alpha: 缩放系数 (推荐 = 2*r)
        target_modules: 注入目标，默认 ["qkv", "proj", "fc1", "fc2"]
        dropout: LoRA dropout
        verbose: 打印注入统计
    Returns:
        model (原地修改)
    """
    if target_modules is None:
        target_modules = ["qkv", "proj", "fc1", "fc2"]

    n = 0
    for block in model.blocks:
        attn = block.attn
        for name in ("qkv", "proj"):
            if name in target_modules and hasattr(attn, name):
                setattr(attn, name, LoRALinear(getattr(attn, name), r, alpha, dropout))
                n += 1
        mlp = block.mlp
        for name in ("fc1", "fc2"):
            if name in target_modules and hasattr(mlp, name):
                setattr(mlp, name, LoRALinear(getattr(mlp, name), r, alpha, dropout))
                n += 1

    for pname, p in model.named_parameters():
        if "lora_" not in pname:
            p.requires_grad = False

    if verbose:
        tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        tot = sum(p.numel() for p in model.parameters())
        print(f"[LoRA] {n} adapters / {len(model.blocks)} blocks | "
              f"trainable {tr:,} / {tot:,} ({100*tr/tot:.2f}%)")
    return model
