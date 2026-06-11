# Copyright (c) Open-CD. All rights reserved.
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from mmengine.structures import PixelData
from mmseg.models.utils import resize
from mmseg.structures import SegDataSample
from opencd.registry import MODELS
from .siamencoder_decoder import SiamEncoderDecoder


@MODELS.register_module()
class ChangeDinoEncoderDecoder(SiamEncoderDecoder):
    """ChangeDino Encoder Decoder segmentor.
    
    组合 ChangeDinoEncoder, ChangeDinoDecoder 和 LearnableSoftMorph。
    兼容 Open-CD 训练和评测流程。

    Args:
        backbone (dict): Configuration for ChangeDinoEncoder
        decode_head (dict): Configuration for ChangeDinoDecoder
        refiner (dict, optional): Configuration for LearnableSoftMorph
        **kwargs: Additional arguments
    """

    def __init__(
        self,
        backbone: dict,
        decode_head: dict,
        refiner: Optional[dict] = None,
        **kwargs
    ):
        # 保存 refiner 配置，先初始化父类
        self.refiner_cfg = refiner
        super().__init__(backbone=backbone, decode_head=decode_head, **kwargs)
        
        # 初始化 Refiner
        if self.refiner_cfg is not None:
            self.refiner = MODELS.build(self.refiner_cfg)
        else:
            self.refiner = None

    def extract_feat(self, inputs: Tensor) -> Tuple[List[Tensor], List[Tensor]]:
        """Extract features from both temporal images.
        
        重写父类方法，返回双时相的特征元组，以匹配 ChangeDinoDecoder 的输入要求。
        
        Args:
            inputs: Input tensor of shape (B, 6, H, W)
            
        Returns:
            Tuple of (feats1, feats2)
        """
        # Split the concatenated inputs into two temporal images
        img_from, img_to = torch.split(inputs, self.backbone_inchannels, dim=1)
        
        # Extract features using the backbone (ChangeDinoEncoder)
        feats1 = self.backbone(img_from)
        feats2 = self.backbone(img_to)
        
        return feats1, feats2

    def encode_decode(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
        """重写父类方法，用于推理时的特征编码解码与 Refiner"""
        feats1, feats2 = self.extract_feat(inputs)
        
        # Decoder 的 predict 返回多尺度的 logits (pred_p2, pred_p3, pred_p4, pred_p5)
        out_tuple = self.decode_head.predict(feats1, feats2, batch_img_metas, self.test_cfg)
        
        # 取最高分辨率的 p2 作为最终输出
        seg_logits = out_tuple[0] 
        
        # Apply refiner if available
        if self.refiner is not None:
            seg_logits = self.refiner(seg_logits)
            
        return seg_logits

    def loss(self, inputs: Tensor, data_samples: List) -> dict:
        """Calculate losses from a batch of inputs and data samples."""
        from mmseg.models.losses import CrossEntropyLoss
        
        feats1, feats2 = self.extract_feat(inputs)
        
        # Get decoder features
        decoder_feats = self.decode_head.forward(feats1, feats2)
        
        # Get predictions at each scale
        pred_p2 = self.decode_head.p2_head(decoder_feats[0])
        pred_p3 = self.decode_head.p3_head(decoder_feats[1])
        pred_p4 = self.decode_head.p4_head(decoder_feats[2])
        pred_p5 = self.decode_head.p5_head(decoder_feats[3])
        
        # Extract ground truth
        gt_label_tensor = torch.cat([ds.gt_sem_seg.data for ds in data_samples]).long()
        if gt_label_tensor.ndim == 4 and gt_label_tensor.shape[1] == 1:
            gt_label_tensor = gt_label_tensor.squeeze(1)
        
        target_size = gt_label_tensor.shape[-2:]
        
        # Resize predictions to match GT
        pred_p2 = resize(pred_p2, size=target_size, mode='bilinear', align_corners=self.align_corners)
        pred_p3 = resize(pred_p3, size=target_size, mode='bilinear', align_corners=self.align_corners)
        pred_p4 = resize(pred_p4, size=target_size, mode='bilinear', align_corners=self.align_corners)
        pred_p5 = resize(pred_p5, size=target_size, mode='bilinear', align_corners=self.align_corners)
        
        # Apply refiner to the highest resolution prediction
        final_pred = pred_p2
        if self.refiner is not None and self.training:
            final_pred = self.refiner(final_pred)
        
        # Loss function with ignore_index=255
        loss_fn = CrossEntropyLoss(
            use_sigmoid=False,
            loss_weight=1.0,
            avg_non_ignore=True,
        )
        
        losses = {}
        losses['loss_ce'] = loss_fn(
            final_pred, 
            gt_label_tensor,
            ignore_index=255  # 🔑 关键：在这里传入
        )
        
        # 辅助损失同理
        aux_weights = {'p3': 0.4, 'p4': 0.3, 'p5': 0.1}
        for name, pred in [('p3', pred_p3), ('p4', pred_p4), ('p5', pred_p5)]:
            if aux_weights[name] > 0:
                losses[f'loss_aux_{name}'] = loss_fn(
                    pred, 
                    gt_label_tensor,
                    ignore_index=255  # 辅助损失也要传入
                ) * aux_weights[name]
        
        return losses


    def predict(self, inputs: Tensor, data_samples: List) -> List:
        """Predict results from a batch of inputs and data samples."""
        if data_samples is not None:
            batch_img_metas = [data_sample.metainfo for data_sample in data_samples]
        else:
            batch_img_metas = [
                dict(ori_shape=inputs.shape[2:], img_shape=inputs.shape[2:], pad_shape=inputs.shape[2:], padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        seg_logits = self.inference(inputs, batch_img_metas)
        return self.postprocess_result(seg_logits, data_samples)

    def _forward(self, inputs: Tensor, data_samples: Optional[List] = None) -> Tensor:
        """Network forward process for tensor mode."""
        feats1, feats2 = self.extract_feat(inputs)
        out_tuple = self.decode_head.forward(feats1, feats2)
        return out_tuple

    def postprocess_result(self, seg_logits: Tensor, data_samples: List) -> List:
        """重写后处理，确保与 MMSeg 格式对齐"""
        batch_size, C, H, W = seg_logits.shape
        if data_samples is None:
            data_samples = [SegDataSample() for _ in range(batch_size)]
            
        for i in range(batch_size):
            i_seg_logits = seg_logits[i]
            
            if C > 1:
                i_seg_pred = i_seg_logits.argmax(dim=0, keepdim=True)
            else:
                i_seg_logits = i_seg_logits.sigmoid()
                i_seg_pred = (i_seg_logits > 0.5).to(i_seg_logits)
                
            data_samples[i].set_data({
                'seg_logits': PixelData(**{'data': i_seg_logits}),
                'pred_sem_seg': PixelData(**{'data': i_seg_pred})
            })
        return data_samples
