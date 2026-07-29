# Copyright (c) Open-CD. All rights reserved.
"""CAU-FLOOD 前光后SAR洪涝变化检测数据集。"""

import os.path as osp

import mmengine

from opencd.registry import DATASETS
from .basecddataset import _BaseCDDataset


@DATASETS.register_module()
class CAUFloodDataset(_BaseCDDataset):
    """CAU-FLOOD 前光后SAR洪涝变化检测数据集（二分类）。

    数据结构::

        data_root/
        ├── train/
        │   ├── opt/          # 前时相光学影像 (3-ch RGB, png)
        │   ├── vv/           # 后时相SAR影像 (1-ch, png)
        │   └── flood_vv/     # 标签 (0=未变化, 1=变化)
        └── test/
            ├── opt/
            ├── vv/
            └── flood_vv/

    无效区域：光学 nodata 或 vv==0，不参与训练和评估。

    Args:
        data_root (str): 数据根目录。
        split (str): 数据集划分，'train' 或 'test'。
        ann_file (str): 样本列表文件，每行一个样本ID（可选）。
        img_suffix (str): 图像后缀名。
        label_suffix (str): 标签文件后缀名。
    """

    METAINFO = dict(
        classes=('unchanged', 'changed'),
        palette=[[0, 0, 0], [255, 255, 255]])

    def __init__(self,
                 data_root='',
                 split='train',
                 ann_file='',
                 img_suffix='.png',
                 label_suffix='',
                 **kwargs):
        self.split = split
        self.img_suffix = img_suffix
        self.label_suffix = label_suffix

        # data_root 和 data_prefix 在 __init__ 中由 super().__init__ 处理
        # 如果调用方未在 data_prefix 中指定路径，这里用默认值补全
        data_prefix = kwargs.get('data_prefix', {})
        if 'img_path_from' not in data_prefix:
            data_prefix['img_path_from'] = osp.join(split, 'opt')
        if 'img_path_to' not in data_prefix:
            data_prefix['img_path_to'] = osp.join(split, 'vv')
        if 'seg_map_path' not in data_prefix:
            data_prefix['seg_map_path'] = osp.join(split, 'flood_vv')
        kwargs['data_prefix'] = data_prefix

        if not ann_file:
            ann_file = ''
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            img_suffix=img_suffix,
            seg_map_suffix=img_suffix,
            **kwargs)

    def load_data_list(self):
        """构建样本列表：img_path=[光学, SAR], seg_map_path=标签。"""
        data_list = []
        pre_dir = self.data_prefix.get('img_path_from', None)
        post_dir = self.data_prefix.get('img_path_to', None)
        label_dir = self.data_prefix.get('seg_map_path', None)

        assert pre_dir is not None and post_dir is not None, \
            '需在 data_prefix 中指定 img_path_from(光学) 与 img_path_to(SAR)'

        # 获取样本 ID
        if self.ann_file and osp.isfile(self.ann_file):
            ids = [line.strip() for line in mmengine.list_from_file(
                self.ann_file, backend_args=self.backend_args) if line.strip()]
        else:
            from mmengine.fileio import list_dir_or_file
            files = sorted(list(list_dir_or_file(
                dir_path=pre_dir, list_dir=False, suffix=self.img_suffix,
                recursive=True, backend_args=self.backend_args)))
            ids = [osp.splitext(osp.basename(f))[0] for f in files]

        for img_id in ids:
            data_info = dict(
                img_path=[
                    osp.join(pre_dir, img_id + self.img_suffix),
                    osp.join(post_dir, img_id + self.img_suffix),
                ])
            if label_dir is not None:
                data_info['seg_map_path'] = osp.join(
                    label_dir, img_id + self.label_suffix + self.img_suffix)
            data_info['label_map'] = self.label_map
            data_info['format_seg_map'] = self.format_seg_map
            data_info['reduce_zero_label'] = self.reduce_zero_label
            data_info['seg_fields'] = []
            data_list.append(data_info)

        return data_list
