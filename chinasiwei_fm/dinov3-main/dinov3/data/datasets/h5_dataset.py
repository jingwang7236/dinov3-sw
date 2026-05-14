import os
import h5py
from typing import Callable, Optional, List, Any, Tuple
import numpy as np
from .decoders import Decoder
from .extended import ExtendedVisionDataset


# 自定义H5解码器（需根据实际H5文件结构调整）
class H5Decoder(Decoder):
    """
    H5文件解码器，支持读取指定数据集（默认'data'）
    """
    def __init__(self, dataset_name: str = "data"):
        self.dataset_name = dataset_name

    def decode(self, data: bytes) -> np.ndarray:
        # 将bytes转为文件对象读取（或直接从路径读取，根据实际场景调整）
        from io import BytesIO
        with h5py.File(BytesIO(data), 'r') as f:
            # 读取指定数据集，可根据需求扩展维度、数据类型转换等
            image = f[self.dataset_name][:]
        return image


# 基础H5数据集加载类
class H5Dataset(ExtendedVisionDataset):
    """
    读取 <root>/list.txt 中的H5文件路径，加载H5格式数据
    每行是H5文件的相对路径或绝对路径
    """
    def __init__(
        self,
        root: str,
        list_file: str = "h5_images.txt",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_decoder: Decoder = H5Decoder,
        h5_dataset_name: str = "data"  # H5文件中存储图像的数据集名称
    ) -> None:
        # 初始化解码器（传入H5数据集名称参数）
        self.h5_decoder = image_decoder(dataset_name=h5_dataset_name)
        super().__init__(
            root=root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            image_decoder=self.h5_decoder
        )
        
        # 读取列表文件，加载H5文件路径
        self.items: List[str] = []
        list_file_path = os.path.join(root, list_file)
        if not os.path.exists(list_file_path):
            raise FileNotFoundError(f"List file {list_file_path} not found!")
        
        with open(list_file_path, "r") as f:
            for ln in f:
                p = ln.strip()
                if not p:
                    continue
                # 支持绝对路径/相对路径
                full_path = p if os.path.isabs(p) else os.path.join(root, p)
                if not os.path.exists(full_path):
                    raise FileNotFoundError(f"H5 file {full_path} not found!")
                self.items.append(full_path)

    def get_image_data(self, index: int) -> bytes:
        """读取H5文件为字节流（供解码器使用）"""
        h5_file_path = self.items[index]
        with open(h5_file_path, "rb") as f:
            return f.read()

    def get_target(self, index: int) -> Any:
        """目标值返回（可根据需求扩展，如从H5读取标签）"""
        return None

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """核心加载逻辑"""
        try:
            # 读取H5字节数据
            image_data = self.get_image_data(index)
            # 解码H5数据为numpy数组
            image = self.image_decoder.decode(image_data)
        except Exception as e:
            error_msg = f"Failed to load H5 file {self.items[index]}"
            print(error_msg)
            raise RuntimeError(f"Can not read H5 image for sample {index}") from e
        
        # 处理目标值（此处无目标值，可扩展）
        target = self.get_target(index)
        target = self.target_decoder(target).decode() if hasattr(self, 'target_decoder') else None

        # 应用数据增强
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.items)

