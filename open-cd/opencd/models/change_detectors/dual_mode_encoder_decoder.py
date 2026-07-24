# Copyright (c) Open-CD. All rights reserved.
"""前光后 SAR 的非对称双流变化检测器。

光学图像走 ``DINOv3AdapterBackbone``（冻结的 DINOv3 ViT，强表征），
SAR 图像走可训练的 ``SARCNNEncoder``（小型 CNN）。两路各自输出
对齐的多尺度特征 [p2, p3, p4, p5]，再交给 decode_head（建议使用
``ChangeDinoCrossAttnDecoder`` 的双向交叉注意力进行异构融合）。

本类继承 ``ChangeDinoEncoderDecoder``，完全复用其 loss / _forward /
refiner / 后处理逻辑，仅覆写 ``__init__``（多建一个 SAR backbone）、
``extract_feat``（两路分别编码）以及 ``encode_decode`` / ``predict``
（支持滑窗推理和 TTA），不修改任何既有模块。
"""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from opencd.registry import MODELS
from .changedino_encoder_decoder import ChangeDinoEncoderDecoder


@MODELS.register_module()
class DualModeBranchEncoderDecoder(ChangeDinoEncoderDecoder):
    """非对称双流 (Optical + SAR) Encoder-Decoder。

    相比 ``ChangeDinoEncoderDecoder``，额外支持:

    1. **滑窗推理** (slide inference): 设 ``test_cfg=dict(mode='slide',
       crop_size=(H,W), stride=(H_s,W_s))``，对大图按窗口滑动推理后
       重叠区域取均值。
    2. **TTA** (Test-Time Augmentation): 设 ``tta_flips`` 参数为翻转
       维度列表，如 ``[3]`` (水平翻转) 或 ``[3, 2]`` (水平+垂直)，
       推理时对每个翻转版本做推理后翻转回来取平均。

    Args:
        backbone_opt (dict): 光学侧 backbone 配置。
        backbone_sar (dict): SAR 侧 backbone 配置。
        decode_head (dict): 解码头配置。
        refiner (dict, optional): 后处理模块配置。
        backbone_inchannels (int): 单张图像的通道数，默认 3。
        tta_flips (list, optional): TTA 翻转维度列表 (基于 [N,C,H,W] 的
            维度索引)。None=不使用 TTA, ``[3]``=水平翻转, ``[3,2]``=水平+垂直。
        ms_inference (bool): 推理时是否做多尺度 logits 加权融合。默认 False
            (仅用 p2, 行为不变, 向后兼容)。True 时把 p2/p3/p4/p5 各尺度
            预测头输出上采样到同尺寸后按 ``ms_inference_weights`` 加权求和,
            常可改善边界与小目标类别。
        ms_inference_weights (list[float], optional): 各尺度 [p2,p3,p4,p5]
            的融合权重。默认 [0.5,0.3,0.1,0.1]。
        **kwargs: 其他透传给父类的参数。
    """

    def __init__(self,
                 backbone_opt: dict,
                 backbone_sar: dict,
                 decode_head: dict,
                 refiner=None,
                 backbone_inchannels: int = 3,
                 tta_flips: Optional[List[int]] = None,
                 ms_inference: bool = False,
                 ms_inference_weights: Optional[List[float]] = None,
                 **kwargs):
        # 光学侧 backbone 作为父类的 self.backbone，复用全部既有逻辑
        super().__init__(
            backbone=backbone_opt,
            decode_head=decode_head,
            refiner=refiner,
            backbone_inchannels=backbone_inchannels,
            **kwargs)
        # SAR 侧 backbone
        self.backbone_sar = MODELS.build(backbone_sar)
        # TTA 配置
        self.tta_flips = tta_flips
        # 多尺度推理融合 (默认关闭, 向后兼容)
        self.ms_inference = ms_inference
        self.ms_inference_weights = ms_inference_weights or [0.5, 0.3, 0.1, 0.1]

    def extract_feat(self, inputs: Tensor) -> Tuple[list, list]:
        """非对称双流特征提取。

        Args:
            inputs (Tensor): 拼接后的双时相输入 [B, 2*C, H, W]，
                前 C 通道为光学，后 C 通道为 SAR。

        Returns:
            (feats_opt, feats_sar): 两路多尺度特征，均为长度 4 的列表
            [p2, p3, p4, p5]，尺度与通道一一对应。
        """
        img_opt, img_sar = torch.split(
            inputs, self.backbone_inchannels, dim=1)
        feats_opt = self.backbone(img_opt)      # 光学 -> DINOv3AdapterBackbone
        feats_sar = self.backbone_sar(img_sar)  # SAR   -> SARCNNEncoder
        return feats_opt, feats_sar

    # ------------------------------------------------------------------
    # encode_decode: 双流编码解码，输出与输入同分辨率的 logits
    # ------------------------------------------------------------------
    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        """双流编码解码，供 slide_inference / whole_inference 调用。

        与父类 ``SiamEncoderDecoder.encode_decode`` 不同，本方法：
        1. 对双流特征分别提取 (feats_opt, feats_sar)；
        2. 直接调用 decoder 内部逻辑获取最高分辨率预测并 resize 到
           **输入尺寸**（而非 ori_shape），使滑窗推理的 crop 输出与
           crop 尺寸严格对齐；
        3. 应用 refiner（如有）。
        """
        feats1, feats2 = self.extract_feat(inputs)
        # 直接取多尺度预测头输出, resize 到输入尺寸
        feats = self.decode_head.extract_feats(feats1, feats2)
        if self.ms_inference:
            seg_logits = self._ms_fuse(feats, inputs.shape[-2:])
        else:
            seg_logits = self.decode_head.p2_head(feats[0])
            if seg_logits.shape[-2:] != inputs.shape[-2:]:
                seg_logits = F.interpolate(
                    seg_logits, size=inputs.shape[-2:], mode='bilinear',
                    align_corners=self.decode_head.align_corners)
        if self.refiner is not None:
            seg_logits = self.refiner(seg_logits)
        return seg_logits

    def _ms_fuse(self, feats: List[Tensor], target_size) -> Tensor:
        """多尺度 logits 加权融合 (仅推理用).

        将 p2/p3/p4/p5 各预测头输出 resize 到 ``target_size`` 后按
        ``self.ms_inference_weights`` 加权求和, 通常可改善边界与小目标。
        """
        heads = [self.decode_head.p2_head, self.decode_head.p3_head,
                 self.decode_head.p4_head, self.decode_head.p5_head]
        weights = self.ms_inference_weights
        align = self.decode_head.align_corners
        out = None
        for head, feat, w in zip(heads, feats, weights):
            logit = head(feat)
            if logit.shape[-2:] != target_size:
                logit = F.interpolate(
                    logit, size=target_size, mode='bilinear',
                    align_corners=align)
            term = logit * w
            out = term if out is None else out + term
        return out

    # ------------------------------------------------------------------
    # predict: 支持滑窗推理 + TTA
    # ------------------------------------------------------------------
    def predict(self, inputs: Tensor,
                data_samples: Optional[List] = None) -> List:
        """推理预测，支持滑窗 (slide) / 整图 (whole) + TTA。

        当 ``test_cfg.mode='slide'`` 时走滑窗推理，否则整图推理。
        当 ``self.tta_flips`` 非空时，对每个翻转版本推理后翻转回来取平均。
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

        # 原始输入推理 (slide 或 whole)
        seg_logits = self.inference(inputs, batch_img_metas)

        # TTA: 翻转增强
        if self.tta_flips:
            for flip_dim in self.tta_flips:
                flipped_inputs = torch.flip(inputs, dims=[flip_dim])
                flipped_logits = self.inference(
                    flipped_inputs, batch_img_metas)
                # 翻转回来再累加
                seg_logits = seg_logits + torch.flip(
                    flipped_logits, dims=[flip_dim])
            seg_logits = seg_logits / (1 + len(self.tta_flips))

        return self.postprocess_result(seg_logits, data_samples)
