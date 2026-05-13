# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Tuple, Union, List
from mmengine.optim import OptimWrapper

import torch
from torch import nn

from mmpretrain.models import BaseSelfSupervisor, CosineEMA
from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample
import random

class FirstArgSequential(nn.Sequential):
    """像 nn.Sequential 一样用，但只把 kwargs 传给第一个模块。"""
    def forward(self, x, **kwargs):
        it = iter(self._modules.values())
        # 第一个子模块接收 kwargs
        out = next(it)(x, **kwargs)
        # 后续模块只接收 out
        for module in it:
            out = module(out)
        return out


@MODELS.register_module()
class DINOHCS(BaseSelfSupervisor):
    """Implementation for DINO.

    This module is proposed in `DINO: Emerging Properties in Self-Supervised
    Vision Transformers <https://arxiv.org/abs/2104.14294>`_.

    Args:
        backbone (dict): Config for backbone.
        neck (dict): Config for neck.
        head (dict): Config for head.
        pretrained (str, optional): Path for pretrained model.
            Defaults to None.
        base_momentum (float, optional): Base momentum for momentum update.
            Defaults to 0.99.
        data_preprocessor (dict, optional): Config for data preprocessor.
            Defaults to None.
        init_cfg (list[dict] | dict, optional): Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 backbone: dict,
                 neck: dict,
                 head: dict,
                 pretrained: Optional[str] = None,
                 base_momentum: float = 0.99,
                 data_preprocessor: Optional[dict] = None,
                 enable_sample: bool = True,
                 init_cfg: Optional[Union[List[dict], dict]] = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            head=head,
            pretrained=pretrained,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

        # create momentum model
        # self.teacher = CosineEMA(
        #     nn.Sequential(self.backbone, self.neck), momentum=base_momentum)
        self.teacher = CosineEMA(
            FirstArgSequential(self.backbone, self.neck), momentum=base_momentum)
            
        # weight normalization layer
        self.neck.last_layer = nn.utils.weight_norm(self.neck.last_layer)
        self.neck.last_layer.weight_g.data.fill_(1)
        self.neck.last_layer.weight_g.requires_grad = False


        self.teacher.module[1].last_layer = nn.utils.weight_norm(
            self.teacher.module[1].last_layer)
        self.teacher.module[1].last_layer.weight_g.data.fill_(1)
        self.teacher.module[1].last_layer.weight_g.requires_grad = False
        # mty add
        self.teacher.module[1].last_layer.weight_v.requires_grad = False
        self.enable_sample = enable_sample

    def loss(self, inputs: torch.Tensor,
             data_samples: List[DataSample]) -> dict:
        global_crops = torch.cat(inputs[:2])
        local_crops = torch.cat(inputs[2:])
        
        if isinstance(data_samples[0], dict):
            channels = [sample['channels'] for sample in data_samples]
        else:
            channels = [sample.channels for sample in data_samples]
        channel = channels[0]
        channels_global = torch.arange(channel, device=global_crops.device).repeat(global_crops.shape[0], 1)
        channels_local = torch.arange(channel, device=local_crops.device).repeat(local_crops.shape[0], 1)

        Cin = channel
        if self.training and self.enable_sample:
            Cin_new = random.randint(1, channel)
            choose_channels = random.sample(range(channel), k=Cin_new)

            extra_global = {"channels": channels_global, "Cin_new": Cin_new, "choose_channels": choose_channels}
            extra_local = {"channels": channels_local, "Cin_new": Cin_new, "choose_channels": choose_channels}
            Cin = Cin_new
        else:
            extra_global = {"channels": channels_global}
            extra_local = {"channels": channels_local}
        # teacher forward
        teacher_output = self.teacher(global_crops, extra_tokens=extra_global)

        # student forward global
        student_output_global = self.backbone(global_crops, extra_global)
        student_output_global = self.neck(student_output_global)

        # student forward local
        student_output_local = self.backbone(local_crops, extra_local)
        student_output_local = self.neck(student_output_local)

        # import ipdb
        # ipdb.set_trace()

        student_output = torch.cat(
            (student_output_global, student_output_local))

        # compute loss
        loss = self.head(student_output, teacher_output)

        return dict(loss=loss, batch_cin=torch.tensor(float(Cin), device=student_output.device))
