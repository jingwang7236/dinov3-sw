# Copyright (c) OpenMMLab. All rights reserved.
import random

from mmcv.transforms import RandomApply  # noqa: E501
from mmcv.transforms import BaseTransform, Compose, RandomFlip, RandomGrayscale

from mmpretrain.datasets.transforms import (ColorJitter, GaussianBlur,
                                            RandomResizedCrop, Solarize)
from mmpretrain.registry import TRANSFORMS


@TRANSFORMS.register_module()
class DINOMultiCrop(BaseTransform):
    """Multi-crop transform for DINO.

    This module applies the multi-crop transform for DINO.

    Args:
        global_crops_scale (int): Scale of global crops.
        local_crops_scale (int): Scale of local crops.
        local_crops_number (int): Number of local crops.
    """

    def __init__(self, global_crops_scale: int, local_crops_scale: int,
                 local_crops_number: int) -> None:
        super().__init__()
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale

        flip_and_color_jitter = Compose([
            RandomFlip(prob=0.5, direction='horizontal'),
            RandomApply([
                ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
            ],
                        prob=0.8),
            RandomGrayscale(
                prob=0.2,
                keep_channels=True,
                channel_weights=(0.114, 0.587, 0.2989),
            )
        ])

        self.global_transform_1 = Compose([
            RandomResizedCrop(
                224,
                crop_ratio_range=global_crops_scale,
                interpolation='bicubic'),
            flip_and_color_jitter,
            GaussianBlur(prob=1.0, radius=random.uniform(0.1, 2.0)),
        ])

        self.global_transform_2 = Compose([
            RandomResizedCrop(
                224,
                crop_ratio_range=global_crops_scale,
                interpolation='bicubic'),
            flip_and_color_jitter,
            GaussianBlur(prob=1.0, radius=random.uniform(0.1, 2.0)),
            Solarize(thr=128, prob=0.2),
        ])

        self.local_crops_number = local_crops_number
        self.local_transform = Compose([
            RandomResizedCrop(
                96,
                crop_ratio_range=local_crops_scale,
                interpolation='bicubic'),
            flip_and_color_jitter,
            GaussianBlur(prob=1.0, radius=random.uniform(0.1, 2.0)),
        ])

    def transform(self, results: dict) -> dict:
        ori_img = results['img']
        crops = []
        results['img'] = ori_img
        crops.append(self.global_transform_1(results)['img'])
        results['img'] = ori_img
        crops.append(self.global_transform_2(results)['img'])
        for _ in range(self.local_crops_number):
            results['img'] = ori_img
            crops.append(self.local_transform(results)['img'])
        results['img'] = crops
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(global_crops_scale = {self.global_crops_scale}, '
        repr_str += f'local_crops_scale = {self.local_crops_scale}, '
        repr_str += f'local_crop_number = {self.local_crops_number})'
        return repr_str




# --- 新增: 多波段/float32 友好的增广实现（cv2 + numpy） ---
import math
from typing import Tuple, Sequence

import cv2
import numpy as np
import torch


def _resize_cv2(img: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    th, tw = size_hw
    return cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR)

class MBRandomResizedCrop:
    """多波段随机裁剪后缩放（不依赖 PIL）。支持 HxW 或 HxWxC，dtype 任意（float32 推荐）"""
    def __init__(self, size: int | Tuple[int, int],
                 scale: Tuple[float, float]=(0.08, 1.0),
                 ratio: Tuple[float, float]=(3/4, 4/3)) -> None:
        self.th, self.tw = (size, size) if isinstance(size, int) else size
        self.scale = scale
        self.ratio = ratio

    def _get_params(self, h, w):
        area = h * w
        log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
        for _ in range(10):
            target = random.uniform(*self.scale) * area
            aspect = math.exp(random.uniform(*log_ratio))
            nh = int(round(math.sqrt(target * aspect)))
            nw = int(round(math.sqrt(target / aspect)))
            if 0 < nh <= h and 0 < nw <= w:
                top = random.randint(0, h - nh)
                left = random.randint(0, w - nw)
                return top, left, nh, nw
        # fallback: 中心裁剪到可接受纵横比
        in_ratio = w / h
        if in_ratio < self.ratio[0]:
            nh = h
            nw = int(round(nh * self.ratio[0]))
        elif in_ratio > self.ratio[1]:
            nw = w
            nh = int(round(nw / self.ratio[1]))
        else:
            nh, nw = h, w
        top = (h - nh) // 2
        left = (w - nw) // 2
        return top, left, nh, nw

    def __call__(self, results: dict) -> dict:
        img = results['img']
        assert img.ndim in (2, 3)
        h, w = img.shape[:2]
        top, left, nh, nw = self._get_params(h, w)
        if img.ndim == 3:
            crop = img[top:top+nh, left:left+nw, :]
        else:
            crop = img[top:top+nh, left:left+nw]
        out = _resize_cv2(crop, (self.th, self.tw))
        results['img'] = out
        return results

class MBRandomFlipH:
    def __init__(self, prob=0.5) -> None:
        self.prob = prob
    def __call__(self, results: dict) -> dict:
        img = results['img']
        if random.random() < self.prob:
            results['img'] = np.ascontiguousarray(np.fliplr(img))
        return results

class MBGaussianBlur:
    """高斯模糊，sigma 随机。对所有通道一次性处理。"""
    def __init__(self, prob=1.0, sigma: Tuple[float, float]=(0.1, 2.0), ksize: int=0) -> None:
        self.prob = prob
        self.sigma = sigma
        self.ksize = ksize
    def __call__(self, results: dict) -> dict:
        if random.random() >= self.prob:
            return results
        img = results['img']
        s = random.uniform(*self.sigma)
        k = int(2 * round(3 * s) + 1) if self.ksize == 0 else (self.ksize | 1)
        results['img'] = cv2.GaussianBlur(img, (k, k), s)
        return results

class MBSolarize:
    """阈值反相（谨慎用于遥感；若值域非[0,1]，请自行改阈值/映射）"""
    def __init__(self, prob=0.2, thr: float=0.5) -> None:
        self.prob = prob
        self.thr = thr
    def __call__(self, results: dict) -> dict:
        if random.random() >= self.prob:
            return results
        img = results['img']
        out = img.copy()
        mask = out > self.thr
        out[mask] = 1.0 - out[mask]  # 如果你的值域不是[0,1]，改成 (max+min - x)
        results['img'] = out
        return results

class MBToCHWFloatTensor:
    """把 HWC/HW ndarray -> CHW float32 Tensor（不归一化）"""
    def __call__(self, results: dict) -> dict:
        img = results['img']
        if img.ndim == 2:
            img = img[:, :, None]
        img = np.ascontiguousarray(img.transpose(2, 0, 1))
        results['img'] = torch.from_numpy(img).float()
        return results

@TRANSFORMS.register_module()
class DINOMultiCropCV2(BaseTransform):
    """多裁剪（cv2/ndarray 版本），支持多波段 float32。"""
    def __init__(self,
                 global_crops_scale: Sequence[float],
                 local_crops_scale: Sequence[float],
                 local_crops_number: int,
                 global_size: int = 224,
                 local_size: int = 96,
                 use_solarize: bool = True) -> None:
        super().__init__()
        self.local_crops_number = local_crops_number

        flip = MBRandomFlipH(prob=0.5)
        # 你可以按需补充“颜色”类增强（多光谱一般不建议强行做 ColorJitter）
        def gblur(p): return MBGaussianBlur(prob=p, sigma=(0.1, 2.0))

        self.global_transform_1 = Compose([
            MBRandomResizedCrop(global_size, scale=tuple(global_crops_scale)),
            flip,
            gblur(1.0),
            MBToCHWFloatTensor(),
        ])
        g2_ops = [
            MBRandomResizedCrop(global_size, scale=tuple(global_crops_scale)),
            flip,
            gblur(1.0),
        ]
        if use_solarize:
            # 注意阈值：若输入未归一化，建议关掉或改成你自己的谱域增强
            g2_ops.append(MBSolarize(prob=0.2, thr=0.5))
        g2_ops.append(MBToCHWFloatTensor())
        self.global_transform_2 = Compose(g2_ops)

        self.local_transform = Compose([
            MBRandomResizedCrop(local_size, scale=tuple(local_crops_scale)),
            flip,
            gblur(1.0),
            MBToCHWFloatTensor(),
        ])

    def transform(self, results: dict) -> dict:
        ori_img = results['img']    # numpy: HxW or HxWxC, float32
        crops = []
        results['img'] = ori_img
        crops.append(self.global_transform_1(results.copy())['img'])
        results['img'] = ori_img
        crops.append(self.global_transform_2(results.copy())['img'])
        for _ in range(self.local_crops_number):
            results['img'] = ori_img
            crops.append(self.local_transform(results.copy())['img'])
        # 输出 list[Tensor(CHW, float32)]
        results['img'] = crops
        return results