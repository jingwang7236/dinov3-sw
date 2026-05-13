import copy
import math
import warnings
from typing import Optional, Sequence, Union, Tuple
import mmengine
import numpy as np
import torch
from einops import einops
from mmengine import to_2tuple, MessageHub, dist
from mmengine.dist import get_dist_info
from mmengine.model import ModuleList, BaseModule
from torch import nn, Tensor
from torch.nn.modules.batchnorm import _BatchNorm
import torch.nn.functional as F
from mmdet.models import FPN as MMDET_FPN
from mmseg.models.backbones.unet import BasicConvBlock
from mmseg.models.utils import UpConvBlock
from opencd.registry import MODELS as MODELS


@MODELS.register_module()
class DualInputsFPN(MMDET_FPN):
    def forward(self, x0: Tuple[Tensor], x1: Tuple[Tensor]) -> (Tuple[Tensor], Tuple[Tensor]):
        bs = x0[0].shape[0]
        x = [torch.cat([x0[i], x1[i]], dim=0) for i in range(len(x0))]
        x = super().forward(x)
        x0 = [i[:bs] for i in x]
        x1 = [i[bs:] for i in x]
        return x0, x1

@MODELS.register_module()
class DualInputsSimpleFusionNeck(BaseModule):
    def __init__(self, in_channels, return_tuple=False, init_cfg=None):
        super().__init__(init_cfg)
        self.in_channels = in_channels
        self.return_tuple = return_tuple
        for i, c in enumerate(in_channels):
            layer = nn.Sequential(
                nn.Conv2d(c * 2, c, 1),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True)
            )
            self.add_module(f'layer{i}', layer)

    def forward(self, x1, x2):
        x = [torch.cat([x1[i], x2[i]], dim=1) for i in range(len(x1))]
        x = [getattr(self, f'layer{i}')(item) for i, item in enumerate(x)]
        if self.return_tuple:
            return (x, )
        return x


@MODELS.register_module()
class UNetDecodeNeck(BaseModule):
    def __init__(self,
                in_channels=[256, 256, 256, 256, 256],
                dec_num_convs=(2, 2, 2, 2),
                dec_dilations=(1, 1, 1, 1),
                with_cp=False,
                conv_cfg=None,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU'),
                upsample_cfg=dict(type='InterpConv'),
                norm_eval=False,
                init_cfg=None):
        super().__init__(init_cfg)
        if init_cfg is None:
            self.init_cfg = [
                dict(type='Kaiming', layer='Conv2d'),
                dict(
                    type='Constant',
                    val=1,
                    layer=['_BatchNorm', 'GroupNorm'])
            ]

        self.norm_eval = norm_eval

        self.in_channels = in_channels
        self.decoder = nn.ModuleList()

        for i in range(len(self.in_channels) - 1):
            self.decoder.append(
                UpConvBlock(
                    conv_block=BasicConvBlock,
                    in_channels=self.in_channels[i + 1],
                    skip_channels=self.in_channels[i],
                    out_channels=self.in_channels[i],
                    num_convs=dec_num_convs[i],
                    stride=1,
                    dilation=dec_dilations[i],
                    with_cp=with_cp,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    upsample_cfg=upsample_cfg,
                    dcn=None,
                    plugins=None))

    def forward(self, enc_outs):
        #  torch.Size([32, 256, 128, 128]), torch.Size([32, 256, 64, 64]), torch.Size([32, 256, 32, 32]), torch.Size([32, 256, 16, 16]), torch.Size([32, 256, 8, 8])
        dec_outs = [enc_outs[-1]]
        x = enc_outs[-1]
        for i in reversed(range(len(self.decoder))):
            x = self.decoder[i](enc_outs[i], x)
            dec_outs.append(x)
        # torch.Size([32, 256, 8, 8]), torch.Size([32, 256, 16, 16]), torch.Size([32, 256, 32, 32]), torch.Size([32, 256, 64, 64]), torch.Size([32, 256, 128, 128])
        return dec_outs

    def train(self, mode=True):
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()