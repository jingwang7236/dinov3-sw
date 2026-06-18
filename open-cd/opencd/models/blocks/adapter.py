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
        # 自动下载权重的地址：/home/users_model/.cache/torch/hub/checkpoints/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth
        # TODO: 暂时无法关闭自动下载，后续尝试使用本地下载的权重
        extract_ids=[5, 11, 17, 23],
        device="cuda",
        freeze_mode: str = "frozen",  # 'frozen', 'unfreeze_last_n', 'full_finetune'
        unfreeze_layers: int = 2
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
