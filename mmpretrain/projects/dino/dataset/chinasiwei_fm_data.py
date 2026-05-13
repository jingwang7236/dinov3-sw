# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Union

from mmengine import fileio
from mmengine.logging import MMLogger

from mmpretrain.registry import DATASETS
from mmpretrain.datasets.custom import CustomDataset


@DATASETS.register_module()
class ChinaSiweiFmDataset(CustomDataset):
    """`ChinaSiweiFmDataset


    Args:
        data_root (str): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to ''.
        split (str): The dataset split, supports "train", "val" and "test".
            Default to ''.
        data_prefix (str | dict): Prefix for training data. Defaults to ''.
        ann_file (str): Annotation file path. Defaults to ''.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        **kwargs: Other keyword arguments in :class:`CustomDataset` and
            :class:`BaseDataset`.


    """  # noqa: E501
    
    
    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tiff', '.tif')
    METAINFO = {'classes': (
        "TreeCover", "Shtubland", "Grassland", "Cropland", "BuiltUp", "BareOrSparseVegetation", "SnowAndIce", "PermanentWaterBodies", "HerbaceousWetlands", "Mangroves", "MossAndLichen"
    )}

    def __init__(self,
                 data_root: str = '',
                 data_prefix: Union[str, dict] = '',
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 **kwargs):
        kwargs = {'extensions': self.IMG_EXTENSIONS, **kwargs}

        # if split:
        #     splits = ['train', 'val', 'test']
        #     assert split in splits, \
        #         f"The split must be one of {splits}, but get '{split}'"

        #     if split == 'test':
        #         logger = MMLogger.get_current_instance()
        #         logger.info(
        #             'Since the ImageNet1k test set does not provide label'
        #             'annotations, `with_label` is set to False')
        #         kwargs['with_label'] = False

        #     data_prefix = split if data_prefix == '' else data_prefix

        #     if ann_file == '':
        #         _ann_path = fileio.join_path(data_root, 'meta', f'{split}.txt')
        #         if fileio.exists(_ann_path):
        #             ann_file = fileio.join_path('meta', f'{split}.txt')

        super().__init__(
            data_root=data_root,
            data_prefix=data_prefix,
            ann_file=ann_file,
            metainfo=metainfo,
            **kwargs)

    def extra_repr(self) -> List[str]:
        """The extra repr information of the dataset."""
        body = [
            f'Root of dataset: \t{self.data_root}',
        ]
        return body

