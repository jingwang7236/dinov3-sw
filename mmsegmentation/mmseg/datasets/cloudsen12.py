# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset
from typing import Callable, Dict, List, Optional, Sequence, Union
import os.path as osp
import os
import mmengine
import mmengine.fileio as fileio


@DATASETS.register_module()
class CloudSen12Dataset(BaseSegDataset):
    METAINFO = dict(
        classes=('clear', 'trick cloud', 'thin cloud', 'cloud shadow'),
        palette=[[0, 0, 0], [255, 255, 255], [255, 255, 0], [61, 68, 71]])

    def __init__(self,
                 img_suffix='.tif',
                 seg_map_suffix='.tif',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)

    def load_data_list(self) -> List[Dict]:
        """Scan `self.data_root` and pair <subdir>/s2l1c.tif with <subdir>/target.tif.

        Expected structure:
            data_root/
            ├─ ROI_xxx_.../
            │   ├─ s2l1c.tif
            │   └─ target.tif
            └─ ROI_xxx_.../
                ├─ s2l1c.tif
                └─ target.tif
        """
        data_list: List[Dict] = []
        root = getattr(self, 'data_root', None) or '.'

        if not osp.isdir(root):
            raise FileNotFoundError(f'data_root not found or not a directory: {root}')

        if osp.isfile(self.ann_file):
            subdirs = []
            lines = mmengine.list_from_file(
                self.ann_file, backend_args=self.backend_args)
            for line in lines:
                folder_name = line.strip()
                subdirs.append(osp.join(root, folder_name))
        else:
            # 可按需改成你希望的排序方式；这里默认按目录名排序，保证可复现
            subdirs = sorted([d for d in os.listdir(root)
                            if osp.isdir(osp.join(root, d))])

        missing_pairs = []

        for sd in subdirs:
            dpath = osp.join(root, sd)
            img_path = osp.join(dpath, 's2l1c_rgb_uint8.tif')
            seg_path = osp.join(dpath, 'target.tif')

            if not osp.isfile(img_path) or not osp.isfile(seg_path):
                missing_pairs.append(sd)
                continue

            info = dict(
                img_path=img_path,
                seg_map_path=seg_path,
                ori_id=sd  # 可选：留作样本ID
            )
            info['label_map'] = self.label_map
            info['reduce_zero_label'] = self.reduce_zero_label
            info['seg_fields'] = []
            data_list.append(info)

        if missing_pairs:
            # 只警告，不中断；你也可以改成 raise
            print(f'[load_data_list] Skipped {len(missing_pairs)} bad folders (missing files): {missing_pairs[:5]}{"..." if len(missing_pairs)>5 else ""}')

        if len(data_list) == 0:
            raise RuntimeError(f'No valid (s2l1c.tif, target.tif) pairs found under {root}')

        return data_list


@DATASETS.register_module()
class ChinasiweiCloudDataset(BaseSegDataset):
    METAINFO = dict(
        classes=('clear', 'trick cloud', 'thin cloud', 'cloud shadow'),
        palette=[[0, 0, 0], [255, 255, 255], [255, 255, 0], [61, 68, 71]])
    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)



@DATASETS.register_module()
class CloudTestDataset(BaseSegDataset):
    METAINFO = dict(
        classes=('clear', 'trick cloud', 'thin cloud', 'cloud shadow'),
        palette=[[0, 0, 0], [255, 255, 255], [255, 255, 0], [61, 68, 71]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
