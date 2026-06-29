# Copyright (c) Open-CD. All rights reserved.
"""DFC2025 BRIGHT 数据集 (前光后 SAR)。

数据结构（光学 / SAR 异构，命名不同）::

    data_root/
    ├── pre-event_wo_ukraine_myanmar_mexico/{id}_pre_disaster.tif   # 光学 (3 通道)
    ├── post-event/{id}_post_disaster.tif                           # SAR   (1 通道)
    ├── target/{id}_building_damage.tif                             # 标签 (0/1/2/3)
    ├── train_set.txt / val_set.txt / test_set.txt                  # 每行一个 {id}

关键事实（已实测）:
1. pre-event 仅含 3395 个样本（与三个 split txt 之和完全一致）；post-event /
   target 多出的 851 个样本来自 mexico-hurricane / myanmar-hurricane /
   ukraine-conflict 三个区域，它们**没有 pre-event 光学图**（目录名
   ``pre-event_wo_ukraine_myanmar_mexico`` 即"不含这三地"），无法用于前光后SAR
   变化检测，故未进入任何 split。本数据集通过 split 文件加载，天然忽略这些多余文件。
2. 标签 ``building_damage.tif`` 是 **4 值**：0=未损毁(背景)，1/2/3=损毁严重程度。
   - 做二分类"变化检测"时需把 1/2/3 合并为 1（默认 ``binarize_label=True``）。
   - 注意：内置的 ``format_seg_map='to_binary'`` 在这里**不可用**，因为它按
     ``<128 -> 0, >=128 -> 1`` 阈值，会把 1/2/3 全部误判为 0。本类通过
     ``label_map={1:1,2:1,3:1}`` 在加载阶段完成合并，无需改动既有 transform。
3. SAR 为单通道；流水线 ``MultiImgLoadImageFromFile`` 默认 ``color_type='color'``
   会自动把灰度复制为 3 通道，与光学侧一致，随后被拼成 6 通道输入。
"""

import os.path as osp

import mmengine

from opencd.registry import DATASETS
from .basecddataset import _BaseCDDataset


@DATASETS.register_module()
class DFC2025BRIGHTDataset(_BaseCDDataset):
    """DFC2025 BRIGHT 数据集（前光后 SAR 变化/损毁检测）。

    Args:
        ann_file (str): split 文件路径（每行一个样本 id，无后缀）。
        pre_suffix / post_suffix / label_suffix (str): 三类文件的命名后缀。
        binarize_label (bool): 是否把损毁等级 1/2/3 合并为 1（二分类变化检测）。
            True 时配合 ``num_classes=2``；False 时为 4 类损毁分级，需配合
            ``num_classes=4``。
        format_seg_map: 固定为 None（标签合并通过 label_map 完成，避免与
            'to_binary' 冲突）。
    """

    METAINFO = dict(
        classes=('unchanged', 'changed'),
        palette=[[0, 0, 0], [255, 255, 255]])

    # 4 值标签 -> 二分类的合并映射
    _BINARIZE_MAP = {1: 1, 2: 1, 3: 1}

    def __init__(self,
                 ann_file: str = '',
                 img_suffix: str = '.tif',
                 seg_map_suffix: str = '.tif',
                 pre_suffix: str = '_pre_disaster',
                 post_suffix: str = '_post_disaster',
                 label_suffix: str = '_building_damage',
                 binarize_label: bool = True,
                 format_seg_map=None,
                 **kwargs) -> None:
        self.pre_suffix = pre_suffix
        self.post_suffix = post_suffix
        self.label_suffix = label_suffix
        self.binarize_label = binarize_label
        super().__init__(
            ann_file=ann_file,
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            format_seg_map=format_seg_map,
            **kwargs)

    def _effective_label_map(self):
        """返回实际写入 data_info 的 label_map。"""
        if self.binarize_label:
            return dict(self._BINARIZE_MAP)
        return self.label_map

    def load_data_list(self):
        """构建样本列表：img_path=[光学, SAR], seg_map_path=标签。"""
        data_list = []
        pre_dir = self.data_prefix.get('img_path_from', None)
        post_dir = self.data_prefix.get('img_path_to', None)
        label_dir = self.data_prefix.get('seg_map_path', None)

        assert pre_dir is not None and post_dir is not None, (
            '需在 data_prefix 中指定 img_path_from(光学) 与 img_path_to(SAR)')

        # 获取样本 id：优先用 split 文件，否则遍历光学目录
        if osp.isfile(self.ann_file):
            ids = [line.strip() for line in mmengine.list_from_file(
                self.ann_file, backend_args=self.backend_args) if line.strip()]
        else:
            from mmengine.fileio import list_dir_or_file
            files = sorted(list(list_dir_or_file(
                dir_path=pre_dir, list_dir=False, suffix=self.img_suffix,
                recursive=True, backend_args=self.backend_args)))
            ids = [osp.splitext(osp.basename(i))[0].rsplit(self.pre_suffix, 1)[0]
                   for i in files]

        label_map = self._effective_label_map()
        for img_id in ids:
            data_info = dict(
                img_path=[
                    osp.join(pre_dir, img_id + self.pre_suffix + self.img_suffix),
                    osp.join(post_dir, img_id + self.post_suffix + self.img_suffix),
                ])
            if label_dir is not None:
                data_info['seg_map_path'] = osp.join(
                    label_dir, img_id + self.label_suffix + self.seg_map_suffix)
            data_info['label_map'] = label_map
            data_info['format_seg_map'] = self.format_seg_map
            data_info['reduce_zero_label'] = self.reduce_zero_label
            data_info['seg_fields'] = []
            data_list.append(data_info)

        return data_list
