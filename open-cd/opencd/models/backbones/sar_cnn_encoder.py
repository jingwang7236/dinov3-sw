# Copyright (c) Open-CD. All rights reserved.
"""可训练的 SAR CNN 编码器。

为前光后 SAR 的非对称双流架构提供 SAR 侧编码器。输出格式与
``DINOv3AdapterBackbone`` 完全对齐：返回 4 个尺度特征列表
[p2, p3, p4, p5]（分别对应 /4, /8, /16, /32），每个特征通道数
为 ``out_channels``，可直接与 ``ChangeDinoDecoder`` /
``ChangeDinoCrossAttnDecoder`` 配合使用。

注：由于 ``opencd/models/backbones`` 目录无写权限，本 backbone 临时
放置于 ``change_detectors`` 目录，但通过 ``MODELS`` 注册表注册，
配置中按 ``type='SARCNNEncoder'`` 引用即可，行为不受影响。
"""

import torch.nn as nn

from opencd.registry import MODELS


def _conv_bn_act(in_c, out_c, stride=1, groups=1, act=nn.ReLU6):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, stride, 1, groups=groups, bias=False),
        nn.BatchNorm2d(out_c),
        act(inplace=True),
    )


class _ResBlock(nn.Module):
    """带残差连接的卷积块。"""

    def __init__(self, channels, act=nn.ReLU6):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels), act(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels))
        self.act = act(inplace=True)

    def forward(self, x):
        return self.act(x + self.conv2(self.conv1(x)))


class _Stage(nn.Module):
    """下采样 + 若干残差块。"""

    def __init__(self, in_c, out_c, stride, n_blocks=2):
        super().__init__()
        self.down = _conv_bn_act(in_c, out_c, stride)
        self.blocks = nn.Sequential(*[_ResBlock(out_c) for _ in range(n_blocks)])

    def forward(self, x):
        return self.blocks(self.down(x))


@MODELS.register_module()
class SARCNNEncoder(nn.Module):
    """可训练的 SAR CNN 编码器。

    结构：stem(/2) -> 4 个 stage，分别输出 /4, /8, /16, /32 的特征，
    再用 1x1 卷积把各尺度投影到 ``out_channels``，与光学侧
    DINOv3AdapterBackbone 的输出尺度、通道一一对应。

    Args:
        in_channels (int): 输入通道数。SAR 若被复制为 3 通道则为 3。
        out_channels (int): 各尺度输出通道数，需与光学侧 / decode_head
            的 ``fpn_channels`` 一致。
        base_channels (int): 基础通道宽度。
        n_blocks (int): 每个 stage 的残差块数量。
    """

    def __init__(self,
                 in_channels=3,
                 out_channels=128,
                 base_channels=32,
                 n_blocks=2,
                 **kwargs):
        super().__init__()
        c = base_channels
        self.stem = _conv_bn_act(in_channels, c, stride=2)              # /2
        self.stage1 = _Stage(c, c, stride=2, n_blocks=n_blocks)         # /4
        self.stage2 = _Stage(c, c * 2, stride=2, n_blocks=n_blocks)     # /8
        self.stage3 = _Stage(c * 2, c * 4, stride=2, n_blocks=n_blocks)  # /16
        self.stage4 = _Stage(c * 4, c * 4, stride=2, n_blocks=n_blocks)  # /32

        self.lateral1 = nn.Conv2d(c, out_channels, 1)
        self.lateral2 = nn.Conv2d(c * 2, out_channels, 1)
        self.lateral3 = nn.Conv2d(c * 4, out_channels, 1)
        self.lateral4 = nn.Conv2d(c * 4, out_channels, 1)

    def forward(self, x):
        x = self.stem(x)
        c1 = self.stage1(x)   # /4
        c2 = self.stage2(c1)  # /8
        c3 = self.stage3(c2)  # /16
        c4 = self.stage4(c3)  # /32
        return [
            self.lateral1(c1), self.lateral2(c2),
            self.lateral3(c3), self.lateral4(c4),
        ]
