import numpy as np

from mmcv.transforms import LoadImageFromFile
from mmdet.registry import TRANSFORMS


# @TRANSFORMS.register_module()
# class LoadImageFromFileSingleBand(LoadImageFromFile):
@TRANSFORMS.register_module()
class LoadImageFromFileSingleBand(BaseTransform):
    """Load an image from file and convert single-band to three-band.
    Inherits from LoadImageFromFile and adds single-band to three-band conversion.
    """
    
    def transform(self, results: dict) -> dict:
        """Functions to load image and convert single-band to three-band.
        
        Args:
            results (dict): Result dict from :obj:`mmcv.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        # 调用父类的transform方法加载图像
        results = super().transform(results)
        
        if results is None:
            return None
        
        img = results['img']
        
        # 关键修改：如果是单波段图像，转换为三波段
        if img.ndim == 2:
            # 单波段图像，复制为三波段
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[2] == 1:
            # 单通道图像（形状为 H,W,1）
            img = np.repeat(img, 3, axis=2)
        
        # 确保图像是三通道
        if img.shape[2] != 3:
            raise ValueError(
                f"Image should have 3 channels after conversion, but got {img.shape[2]} channels")
        
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]

        import pdb
        pdb.set_trace()

        return results