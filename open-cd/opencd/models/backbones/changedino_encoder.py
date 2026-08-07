import timm 

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from opencd.models.blocks import mobilenet_v2, FPN, DINOV3Wrapper, DenseAdapterLite, PyramidFeatureFusion

from opencd.registry import MODELS


def get_backbone(backbone_name, mobilenet_pretrained=None):
    if backbone_name == "mobilenetv2":
        backbone = mobilenet_v2(pretrained=False, progress=True)
        if mobilenet_pretrained:
            if os.path.isfile(mobilenet_pretrained):
                state_dict = torch.load(
                    mobilenet_pretrained, map_location="cpu")
                msg = backbone.load_state_dict(state_dict, strict=False)
                print(f"loading local imagenet pretrained mobilenetv2: "
                      f"{mobilenet_pretrained} "
                      f"(missing={len(msg.missing_keys)}, "
                      f"unexpected={len(msg.unexpected_keys)})")
            else:
                print(f"[WARNING] mobilenet_v2 ImageNet weights not found at "
                      f"{mobilenet_pretrained}; using random init.")
        else:
            print("[WARNING] mobilenet_pretrained not set; "
                  "mobilenetv2 uses random init.")
        backbone.channels = [16, 24, 32, 96, 320]
    elif backbone_name == "resnet18d":
        backbone = timm.create_model("resnet18d", pretrained=False, features_only=True)
        backbone.channels = [64, 64, 128, 256, 512]
    else:
        raise NotImplementedError("BACKBONE [%s] is not implemented!\n" % backbone_name)
    return backbone


@MODELS.register_module()
class ChangeDinoEncoder(nn.Module):
    def __init__(
        self,
        backbone="mobilenetv2",
        fpn_channels=128,
        deform_groups=4,
        gamma_mode="SE",
        beta_mode="contextgatedconv",
        dino_weight="/mnt/ht2-nas2/00-model/00-wj/Codes/checkpoints/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
        weights_type="auto",
        device="cuda",
        extract_ids=[5, 11, 17, 23],
        mobilenet_pretrained=None,
        freeze_mode="frozen",
        unfreeze_last_n=0,
        **kwargs,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.backbone = get_backbone(backbone, mobilenet_pretrained)
        self.fpn = FPN(
            in_channels=self.backbone.channels[-4:],
            out_channels=fpn_channels,
            deform_groups=deform_groups,
            gamma_mode=gamma_mode,
            beta_mode=beta_mode,
        )
        dense_out_dim = fpn_channels * 2
        self.dino = DINOV3Wrapper(
            weights_path=dino_weight,
            device=device,
            extract_ids=extract_ids,
            weights_type=weights_type,
            freeze_mode=freeze_mode,
            unfreeze_layers=unfreeze_last_n,
        )
        self.dense_adp = DenseAdapterLite(
            in_dim=1024, out_dim=dense_out_dim, bottleneck=fpn_channels // 2
        )
        self.pff = PyramidFeatureFusion(
            in_dims=[fpn_channels] * 4,
            dense_dim=1024,
            patch_size=self.dino.patch_size,
            hidden_dim=dense_out_dim,
        )

    def forward(self, x):
        """
        x1: [B, 3, H, W]
        x2: [B, 3, H, W]
        return: [B, 1, H, W]
        """
        fea = self.backbone.forward(x)
        fea = self.fpn(fea[-4:])  # t1_p2, t1_p3, t1_p4, t1_p5

        ds_fea = self.dino(x)

        # process dense features
        ds_fea = self.dense_adp(ds_fea)

        fea = self.pff(fea, ds_fea)

        return fea

    def set_freeze_mode(self, mode="frozen", n=0):
        """动态切换 DINO 分支冻结模式 (兼容 FreezeScheduleHook)。

        仅作用于 self.dino (DINOV3Wrapper)，CNN 主干 / FPN / adapter 始终可训练。
        """
        self.dino.set_freeze_mode(mode=mode, n=n)

@MODELS.register_module()
class ChangeDinoEncoderOnlyDino(nn.Module):
    def __init__(
        self,
        out_channels=128,
        dino_weight="/mnt/ht2-nas2/00-model/00-wj/Codes/checkpoints/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
        device="cuda",
        extract_ids=[5, 11, 17, 23],
        freeze_mode="frozen",
        unfreeze_layers=2,
        **kwargs,
    ):
        super().__init__()
        # 显式透传 freeze_mode / unfreeze_layers，保证配置中的训练策略真正生效
        self.dino = DINOV3Wrapper(
            weights_path=dino_weight,
            device=device,
            extract_ids=extract_ids,
            freeze_mode=freeze_mode,
            unfreeze_layers=unfreeze_layers,
        )
        self.dense_adp = DenseAdapterLite(
            in_dim=1024, out_dim=out_channels, bottleneck=out_channels // 2
        )

    def forward(self, x):
        ds_fea = self.dino(x)
        ds_fea = self.dense_adp(ds_fea)
        return ds_fea


# ================================================================
# LoRA 插件式 Encoder — 继承 ChangeDinoEncoderOnlyDino，不修改原有类
# 配置中将 type 改为 'ChangeDinoEncoderLoRA' 即可启用 LoRA
# ================================================================
from opencd.models.blocks.adapter import apply_lora


@MODELS.register_module()
class ChangeDinoEncoderLoRA(ChangeDinoEncoderOnlyDino):
    """ChangeDino Encoder + LoRA 插件。

    继承 ChangeDinoEncoderOnlyDino 的全部行为，在初始化后自动注入 LoRA：
    1. 父类 __init__ 以 frozen 模式加载 DINOv3 权重并冻结
    2. 将 freeze_mode 改为非 frozen（使 forward 不再使用 no_grad，梯度可流经 LoRA）
    3. 注入 LoRA 适配器（原始权重保持冻结，仅 lora_A/lora_B 可训练）

    Config 示例::

        backbone=dict(
            type='ChangeDinoEncoderLoRA',
            out_channels=128,
            extract_ids=[5, 11, 17, 23],
            dino_weight='...',
            lora_r=8,
            lora_alpha=16,
            lora_target_modules=["qkv", "proj", "fc1", "fc2"],
        )
    """

    def __init__(
        self,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        lora_target_modules: list = None,
        **kwargs,
    ):
        # 父类以 frozen 模式初始化（加载权重 + 冻结全部参数）
        kwargs.setdefault("freeze_mode", "frozen")
        super().__init__(**kwargs)

        # 关键：将 freeze_mode 设为非 frozen，使 DINOV3Wrapper.forward
        # 使用 nullcontext() 而非 no_grad()，梯度才能流经冻结层到达 LoRA
        self.dino.freeze_mode = "lora"

        # 注入 LoRA 适配器（原始权重保持 requires_grad=False）
        apply_lora(
            self.dino.model,
            r=lora_r,
            alpha=lora_alpha,
            target_modules=lora_target_modules,
            dropout=lora_dropout,
        )