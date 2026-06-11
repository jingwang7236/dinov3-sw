import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union

from opencd.models.blocks import TransformerBlock
from opencd.registry import MODELS


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
                x1, size=x2.shape[-2:], 
                mode="bilinear", 
                align_corners=False
            )
        g = self.gate(torch.cat([x1, x2], dim=1))
        fused = x2 + g * x1
        return self.mix(fused)


@MODELS.register_module()
class ChangeDinoDecoder(nn.Module):
    """
    ChangeDino Decoder for SiamEncoderDecoder.
    
    兼容父类 SiamEncoderDecoder 所需的属性：
        - align_corners: 插值时是否对齐角点
        - num_classes: 分割类别数
        - out_channels: 输出通道数（等于 num_classes）
        - loss: 计算损失的方法
        - predict: 推理方法
    
    Args:
        fpn_channels (int): Number of channels from FPN backbone.
        n_layers (List[int]): Number of transformer blocks at each scale.
        num_classes (int): Number of segmentation classes (default: 2).
        align_corners (bool): Align corners in interpolation (default: False).
        decode_head (dict, optional): Additional decode head config.
        **kwargs: Additional arguments.
    """
    
    def __init__(
        self,
        fpn_channels: int = 128,
        n_layers: List[int] = [1, 1, 1, 1],
        num_classes: int = 2,
        align_corners: bool = False,
        **kwargs,
    ):
        super().__init__()
        
        # ========== 父类要求的属性 ==========
        self.align_corners = align_corners
        self.num_classes = num_classes
        self.out_channels = num_classes
        
        # ========== 模型参数 ==========
        self.fpn_channels = fpn_channels
        self.n_layers = n_layers
        
        # ========== 特征融合模块 ==========
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
        self, 
        dim: int, 
        attn_type: str, 
        num_blocks: int, 
        depth: int,
        **extra_kwargs
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
    
    def forward(self, x1s, x2s, size=None):
        """
        前向传播，返回多尺度特征（用于 extract_feat）。
        
        Args:
            x1s: Features from time 1 (p2, p3, p4, p5)
            x2s: Features from time 2 (p2, p3, p4, p5)
        
        Returns:
            List of multi-scale feature tensors [feat_p2, feat_p3, feat_p4, feat_p5]
        """
        # 解包特征
        t1_p2, t1_p3, t1_p4, t1_p5 = x1s
        t2_p2, t2_p3, t2_p4, t2_p5 = x2s
        
        # 计算差异特征
        diff_p5 = self._compute_diff(t1_p5, t2_p5)
        diff_p4 = self._compute_diff(t1_p4, t2_p4)
        diff_p3 = self._compute_diff(t1_p3, t2_p3)
        diff_p2 = self._compute_diff(t1_p2, t2_p2)
        
        # 自顶向下处理
        feat_p5 = self.tb5(diff_p5)
        feat_p4 = self.p5_to_p4(feat_p5, diff_p4)
        feat_p4 = self.tb4(feat_p4)
        feat_p3 = self.p4_to_p3(feat_p4, diff_p3)
        feat_p3 = self.tb3(feat_p3)
        feat_p2 = self.p3_to_p2(feat_p3, diff_p2)
        feat_p2 = self.tb2(feat_p2)
        
        # 返回多尺度特征（用于后续 decode_head 的 forward）
        return [feat_p2, feat_p3, feat_p4, feat_p5]
    
    def loss(self, inputs: List[torch.Tensor], data_samples: List, train_cfg: Optional[dict] = None) -> dict:
        """
        计算损失（父类 SiamEncoderDecoder 要求的方法）。
        
        Args:
            inputs: 来自 backbone 的多尺度特征 [feat_p2, feat_p3, feat_p4, feat_p5]
            data_samples: 数据样本列表，包含 ground truth
            train_cfg: 训练配置（可选）
        
        Returns:
            损失字典
        """
        from mmseg.models import build_loss
        from mmseg.models.losses import CrossEntropyLoss
        
        # 获取输入特征
        feat_p2, feat_p3, feat_p4, feat_p5 = inputs
        
        # 计算各尺度的预测
        pred_p5 = self.p5_head(feat_p5)
        pred_p4 = self.p4_head(feat_p4)
        pred_p3 = self.p3_head(feat_p3)
        pred_p2 = self.p2_head(feat_p2)
        
        # 获取目标尺寸（与输入图像一致）
        target_size = data_samples[0].gt_sem_seg.shape[-2:] if data_samples else (512, 512)
        
        # 上采样到目标尺寸
        pred_p2 = F.interpolate(pred_p2, size=target_size, mode="bilinear", align_corners=self.align_corners)
        pred_p3 = F.interpolate(pred_p3, size=target_size, mode="bilinear", align_corners=self.align_corners)
        pred_p4 = F.interpolate(pred_p4, size=target_size, mode="bilinear", align_corners=self.align_corners)
        pred_p5 = F.interpolate(pred_p5, size=target_size, mode="bilinear", align_corners=self.align_corners)
        
        # 使用 CrossEntropyLoss 计算损失
        loss_fn = CrossEntropyLoss(use_sigmoid=False, loss_weight=1.0)
        
        losses = {}
        
        # 主损失（使用最高分辨率 p2）
        losses['loss_ce'] = loss_fn(pred_p2, data_samples)
        
        # 辅助损失（深度监督）
        aux_weights = {'p3': 0.5, 'p4': 0.3, 'p5': 0.1}
        for name, pred, weight in [('p3', pred_p3, 0.5), ('p4', pred_p4, 0.3), ('p5', pred_p5, 0.1)]:
            if weight > 0:
                losses[f'loss_aux_{name}'] = loss_fn(pred, data_samples) * weight
        
        return losses
    
    def predict(self, inputs: List[torch.Tensor], batch_img_metas: List[dict], test_cfg: Optional[dict] = None) -> torch.Tensor:
        """
        推理方法（父类 SiamEncoderDecoder 要求的方法）。
        
        Args:
            inputs: 来自 backbone 的多尺度特征 [feat_p2, feat_p3, feat_p4, feat_p5]
            batch_img_metas: 图像元信息列表
            test_cfg: 测试配置（可选）
        
        Returns:
            seg_logits: 分割 logits，形状 (N, C, H, W)
        """
        # 获取输入特征
        feat_p2, feat_p3, feat_p4, feat_p5 = inputs
        
        # 计算各尺度的预测
        pred_p5 = self.p5_head(feat_p5)
        pred_p4 = self.p4_head(feat_p4)
        pred_p3 = self.p3_head(feat_p3)
        pred_p2 = self.p2_head(feat_p2)
        
        # 获取目标尺寸（从图像元信息中获取）
        ori_shape = batch_img_metas[0]['ori_shape']
        target_size = ori_shape
        
        # 上采样到原始图像尺寸
        pred_p2 = F.interpolate(pred_p2, size=target_size, mode="bilinear", align_corners=self.align_corners)
        pred_p3 = F.interpolate(pred_p3, size=target_size, mode="bilinear", align_corners=self.align_corners)
        pred_p4 = F.interpolate(pred_p4, size=target_size, mode="bilinear", align_corners=self.align_corners)
        pred_p5 = F.interpolate(pred_p5, size=target_size, mode="bilinear", align_corners=self.align_corners)
        
        # 返回最高分辨率的预测作为最终输出
        return pred_p2,pred_p3,pred_p4,pred_p5