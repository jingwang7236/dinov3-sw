import os
from typing import Callable, Optional, List, Any
from .decoders import Decoder, MultiBandTiffDecoder, ImageDataDecoder, TargetDecoder, ChannelSelectTIFFDecoder
from .extended import ExtendedVisionDataset
from typing import Any, Tuple
import numpy as np
import tifffile
import random
from PIL import Image
class ChinasiweiDataset(ExtendedVisionDataset):
    """
    读取 <root>/list.txt，每行是图片相对路径或绝对路径。
    """
    def __init__(
        self,
        root: str,
        list_file: str = "3bands_images.txt",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        # image_decoder: Decoder = ImageDataDecoder,
        image_decoder: Decoder = MultiBandTiffDecoder,
        crop_size: int = 512,
    ) -> None:
        super().__init__(root=root, transforms=transforms, transform=transform,
                         target_transform=target_transform, image_decoder=image_decoder)
        self.items: List[str] = []
        self.crop_size = crop_size
        with open(os.path.join(root, list_file), "r") as f:
            for ln in f:
                p = ln.strip()
                if not p:
                    continue
                self.items.append(p)

    def get_image_data(self, index: int) -> bytes:
        full = os.path.join(self.root, self.items[index])
        with open(full, "rb") as f:
            return f.read()

    def get_target(self, index: int) -> Any:
        return None

    # def __getitem__(self, index: int) -> Tuple[Any, Any]:
    #     try:
    #         image_data = self.get_image_data(index)
    #         image = self.image_decoder(image_data).decode()
    #     except Exception as e:
    #         print(f"failed to load {self.items[index]}")
    #         raise RuntimeError(f"can not read image for sample {index}") from e
    #     target = self.get_target(index)
    #     target = self.target_decoder(target).decode()

    #     if self.transforms is not None:
    #         image, target = self.transforms(image, target)

    #     return image, target
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path = os.path.join(self.root, self.items[index])
        
        try:
            # 获取图像尺寸 (tifffile 读取元数据很快，不加载像素数据)
            with tifffile.TiffFile(path) as tif:
                page = tif.pages[0]
                img_h, img_w = page.shape[0], page.shape[1]
                if img_w > self.crop_size:
                    x_start = random.randint(0, img_w - self.crop_size)
                else:
                    x_start = 0
                    
                if img_h > self.crop_size:
                    y_start = random.randint(0, img_h - self.crop_size)
                else:
                    y_start = 0
                
                # 只读取窗口区域的数据 (Out-of-core reading)
                # key='contiguous' 确保数据连续存储，提高读取速度
                window_data = page.asarray(key='contiguous')[
                    y_start:y_start+self.crop_size, 
                    x_start:x_start+self.crop_size
                ]
                
                # window_data shape: (H, W, C) or (H, W) depending on TIFF structure
                # 确保是 HWC 格式
                if window_data.ndim == 3 and window_data.shape[0] < 10: # 可能是 CHW
                     window_data = np.transpose(window_data, (1, 2, 0))
                
                # 数据类型转换与归一化
                # 假设是 uint16 遥感数据，转换为 float32 并归一化到 0-1 或标准分布
                if window_data.dtype == np.uint16:
                    window_data = window_data.astype(np.float32) / 65535.0
                elif window_data.dtype == np.uint8:
                    window_data = window_data.astype(np.float32) / 255.0
                
                # 转换为 PIL Image 以兼容后续的 DINO transforms
                # 如果波段数 > 3，取前三个波段用于可视化/预训练，或者根据需求处理
                if window_data.shape[-1] >= 3:
                    pil_img = Image.fromarray((window_data[:, :, :3] * 255).astype(np.uint8))
                else:
                    # 如果是单波段，复制成三波段
                    gray = (window_data * 255).astype(np.uint8)
                    pil_img = Image.merge("RGB", [Image.fromarray(gray), Image.fromarray(gray), Image.fromarray(gray)])

        except Exception as e:
            print(f"Failed to load {path}: {e}")
            # 返回一个空白图像避免训练中断，或者抛出异常
            pil_img = Image.new("RGB", (self.crop_size, self.crop_size), (0, 0, 0))

        target = self.get_target(index)
        target = self.target_decoder(target).decode()
        
        if self.transforms is not None:
            pil_img, target = self.transforms(pil_img, target)

        return pil_img, target
    def __len__(self) -> int:
        return len(self.items)


def build_channel_first_index(image_paths, n_bands):
    image_paths = np.asarray(image_paths, dtype=object)
    n_bands = np.asarray(n_bands, dtype=np.int16)
    maxC = int(n_bands.max()) if len(n_bands) else 0

    img_ids_list = []
    band_list = []

    for b in range(maxC):
        ids = np.flatnonzero(n_bands > b) 
        img_ids_list.append(ids)
        band_list.append(np.full(ids.shape[0], b, dtype=np.int16))

    img_ids = np.concatenate(img_ids_list) if img_ids_list else np.empty(0, np.int64)
    bands = np.concatenate(band_list) if band_list else np.empty(0, np.int16)

    expanded_paths = image_paths[img_ids]
    return expanded_paths.tolist(), np.expand_dims(bands, 1).tolist()


class ChinasiweiBoCDataset(ExtendedVisionDataset):
    def __init__(
        self,
        root: str,
        list_file: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_decoder: Decoder = ChannelSelectTIFFDecoder,
    ) -> None:
        super().__init__(root=root, transforms=transforms, transform=transform,
                         target_transform=target_transform, image_decoder=image_decoder)
        self.root = root
        self.channel_adaptive = True

        self._image_paths = []
        self._n_channels = []
        with open(os.path.join(root, list_file), "r") as f:
            for ln in f:
                p = ln.strip()
                if not p:
                    continue
                channel, path = p.split(', ')
                self._image_paths.append(path)
                self._n_channels.append(channel)

        self._image_paths, self._channels = build_channel_first_index(self._image_paths, self._n_channels)
  
    def get_image_data(self, index: int) -> bytes:
        full = os.path.join(self.root, self._image_paths[index])
        with open(full, "rb") as f:
            image_data = f.read()
        if self.channel_adaptive:
            channels = self._channels[index]
            image_data = image_data + bytes(channels) + (len(channels)).to_bytes(1, byteorder="big")

            return image_data
        else:
            return image_data


    def get_target(self, index: int) -> Any:
        return None

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        try:
            image_data = self.get_image_data(index)
            image = self.image_decoder(image_data).decode()
        except Exception as e:
            print(f"failed to load {self.items[index]}")
            raise RuntimeError(f"can not read image for sample {index}") from e
        target = self.get_target(index)
        target = self.target_decoder(target).decode()

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self._image_paths)
