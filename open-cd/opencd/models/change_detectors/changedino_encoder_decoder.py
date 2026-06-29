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

    组合 SiameseDinoV3Backbone + ChangeDinoDecoder + LearnableSoftMorph。
    兼容 Open-CD 训练和评测流程。

    架构流程:
        1. extract_feat: backbone 分别提取双时相特征 -> (feats1, feats2)
        2. loss:         decode_head.loss(feats1, feats2, data_samples)
        3. predict:      decode_head.predict(feats1, feats2, batch_img_metas) + refiner
        4. _forward:     decode_head._forward(feats1, feats2)

    Args:
        backbone (dict): Backbone 配置 (SiameseDinoV3Backbone)
        decode_head (dict): Decoder 配置
        refiner (dict, optional): LearnableSoftMorph 配置
        **kwargs: 其他参数
    """

    def __init__(
        self,
        backbone: dict,
        decode_head: dict,
        refiner: Optional[dict] = None,
        **kwargs,
    ):
        self.refiner_cfg = refiner
        super().__init__(backbone=backbone, decode_head=decode_head, **kwargs)

        # 初始化 Refiner
        self.refiner = MODELS.build(self.refiner_cfg) if self.refiner_cfg else None

    # ------------------------------------------------------------------
    # extract_feat: 拆分双时相输入，分别提取特征
    # ------------------------------------------------------------------
    def extract_feat(
        self, inputs: Tensor
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """提取双时相特征。

        Args:
            inputs: 拼接后的输入张量

        Returns:
            (feats1, feats2): 双时相的多尺度特征
        """
        img_from, img_to = torch.split(
            inputs, self.backbone_inchannels, dim=1
        )
        feats1 = self.backbone(img_from)
        feats2 = self.backbone(img_to)
        return feats1, feats2

    # ------------------------------------------------------------------
    # loss: 训练损失计算，委托给 decode_head.loss
    # ------------------------------------------------------------------
    def loss(self, inputs: Tensor, data_samples: List) -> dict:
        """计算训练损失。

        与官方 ChangeDINO 对齐：
        - refiner 输出 final_pred 单独以全权重参与 focal/dice，并正常反传训练 refiner
        - 辅助分支 p2/p3/p4/p5 的 focal 权重为 1.0，dice 权重为 0.5
        - 总损失 = 0.5 * focal + dice

        Args:
            inputs: 输入张量
            data_samples: 数据样本列表

        Returns:
            损失字典
        """
        feats1, feats2 = self.extract_feat(inputs)
        # 1. 解码器多尺度预测 + 辅助损失
        # loss_with_preds 内部已对 p2/p3/p4/p5 计算 focal(×1.0)+dice(×0.5)
        loss_dict, preds = self.decode_head.loss_with_preds(
            feats1, feats2, data_samples
        )

        # 2. 解析 GT
        gt_label_tensor = self.decode_head._parse_gt(data_samples)

        # 3. 应用 Refiner 并以全权重计算损失（关键：使 refiner 参与训练）
        if self.refiner is not None:
            final_pred = preds[0]  # pred_p2 (已上采样到 GT 尺寸)
            final_pred = self.refiner(final_pred)
            if final_pred.shape[-2:] != gt_label_tensor.shape[-2:]:
                final_pred = F.interpolate(
                    final_pred,
                    size=gt_label_tensor.shape[-2:],
                    mode='bilinear',
                    align_corners=self.decode_head.align_corners
                )
            ref_focal = self.decode_head.focal_loss(
                final_pred, gt_label_tensor)
            ref_dice = self.decode_head.dice_loss(
                final_pred, gt_label_tensor)
            # refiner 输出：focal/dice 均为全权重(1.0)
            loss_dict['loss_focal'] = loss_dict['loss_focal'] + ref_focal
            loss_dict['loss_dice'] = loss_dict['loss_dice'] + ref_dice
            # Lovász (可选)
            if self.decode_head.lovasz_loss is not None:
                ref_lovasz = self.decode_head.lovasz_loss(
                    final_pred, gt_label_tensor)
                if 'loss_lovasz' in loss_dict:
                    loss_dict['loss_lovasz'] = (
                        loss_dict['loss_lovasz']
                        + ref_lovasz * self.decode_head.lovasz_weight)
                else:
                    loss_dict['loss_lovasz'] = (
                        ref_lovasz * self.decode_head.lovasz_weight)

        # 4. 官方总损失 = 0.5 * focal + dice
        if 'loss_focal' in loss_dict:
            loss_dict['loss_focal'] = loss_dict['loss_focal'] * 0.5

        return loss_dict

    # ------------------------------------------------------------------
    # predict: 推理预测，委托给 decode_head.predict + refiner
    # ------------------------------------------------------------------
    def predict(self, inputs: Tensor, data_samples: List) -> List:
        """推理预测。

        Args:
            inputs: 输入张量
            data_samples: 数据样本列表

        Returns:
            后处理后的数据样本列表
        """
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0],
                )
            ] * inputs.shape[0]

        # 1. 提取特征
        feats1, feats2 = self.extract_feat(inputs)

        # 2. Decoder 预测 -> seg_logits (N, C, H, W)
        seg_logits = self.decode_head.predict(feats1, feats2, batch_img_metas)

        # 3. 应用 Refiner
        if self.refiner is not None:
            seg_logits = self.refiner(seg_logits)

        # 4. 后处理
        return self.postprocess_result(seg_logits, data_samples)

    # ------------------------------------------------------------------
    # _forward: 纯前向
    # ------------------------------------------------------------------
    def _forward(
        self, inputs: Tensor, data_samples: Optional[List] = None
    ) -> Tensor:
        """纯前向传播。

        Args:
            inputs: 输入张量
            data_samples: 数据样本列表（未使用）

        Returns:
            多尺度预测 logits
        """
        feats1, feats2 = self.extract_feat(inputs)
        # return self.decode_head._forward(feats1, feats2)
        preds = self.decode_head._forward(feats1, feats2)
        
        # 提取最高分辨率预测
        if isinstance(preds, (tuple, list)):
            final_pred = preds[0]
        else:
            final_pred = preds
        
        # 应用 Refiner
        if self.refiner is not None:
            final_pred = self.refiner(final_pred)
        
        return final_pred

    # ------------------------------------------------------------------
    def postprocess_result(
        self, seg_logits: Tensor, data_samples: List
    ) -> List:
        """后处理，将 logits 转为预测结果。

        Args:
            seg_logits: (N, C, H, W)
            data_samples: 数据样本列表

        Returns:
            更新后的数据样本列表
        """
        batch_size, C, H, W = seg_logits.shape
        if data_samples is None:
            data_samples = [SegDataSample() for _ in range(batch_size)]

        for i in range(batch_size):
            i_seg_logits = seg_logits[i]
            # 获取原始图像尺寸
            ori_shape = data_samples[i].metainfo.get('ori_shape', (H, W))
            
            # 如果尺寸不匹配，上采样到原始尺寸
            i_seg_logits = seg_logits[i]
            if i_seg_logits.shape[-2:] != ori_shape:
                i_seg_logits = F.interpolate(
                    i_seg_logits.unsqueeze(0), 
                    size=ori_shape, 
                    mode='bilinear', 
                    align_corners=self.align_corners
                ).squeeze(0)
                
            if C > 1:
                i_seg_pred = i_seg_logits.argmax(dim=0, keepdim=True)
            else:
                i_seg_logits = i_seg_logits.sigmoid()
                i_seg_pred = (i_seg_logits > 0.5).to(i_seg_logits)

            data_samples[i].set_data({
                'seg_logits': PixelData(**{'data': i_seg_logits}),
                'pred_sem_seg': PixelData(**{'data': i_seg_pred}),
            })

        return data_samples