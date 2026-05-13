# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class Muti_Class_Dataset(BaseSegDataset):
    """Muti_Class_Dataset dataset.

    In segmentation map annotation for LoveDA, 0 is the ignore index.
    ``reduce_zero_label`` should be set to True. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.png'.
    """
    METAINFO = dict(
        # classes=('耕地', '草地', '林地', '灌木林', '建筑物', '水体',
        #          '道路', '冰雪', '裸地', '构筑物'),

        classes=('Cropland', 'Grassland', 'Forest', 'Shrubland', 'Buildings', 'Water',
                 'Road', 'Snow', 'bare_land', 'Structures'),
        palette=[[255, 235, 175], [233, 255, 190], [0, 100, 0], [255, 190, 35],
                 [255, 170, 0], [0, 100, 200], [255, 0, 0], [240, 240, 240],
                 [250, 230, 160], [0, 150, 160]])

    def __init__(self,
                 img_suffix='.tif',
                 seg_map_suffix='.png',
                 reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
