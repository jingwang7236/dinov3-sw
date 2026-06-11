import os
import gc
from typing import Callable, Optional, List, Any
from .decoders import Decoder, MultiBandTiffDecoder, ImageDataDecoder, TargetDecoder, ChannelSelectTIFFDecoder
from .extended import ExtendedVisionDataset
from typing import Any, Tuple
import numpy as np
import random
from PIL import Image
import rasterio
from rasterio.windows import Window
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
        self.iter_count = 0

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
        pil_img = None
        try:
            # 1. 打开文件获取尺寸和波段信息
            with rasterio.open(path) as src:
                img_h, img_w = src.height, src.width
                num_bands = src.count
                
                # 2. 计算随机裁剪窗口
                if img_w > self.crop_size:
                    x_start = random.randint(0, img_w - self.crop_size)
                else:
                    x_start = 0
                    
                if img_h > self.crop_size:
                    y_start = random.randint(0, img_h - self.crop_size)
                else:
                    y_start = 0
                
                window = Window(x_start, y_start, self.crop_size, self.crop_size)
                
                # 3. 【关键修改】根据波段数动态读取
                if num_bands >= 3:
                    # 读取前三个波段
                    indexes = (1, 2, 3)
                    window_data = src.read(indexes=indexes, window=window, out_dtype='float32')
                else:
                    # 读取所有可用波段
                    indexes = tuple(range(1, num_bands + 1))
                    window_data = src.read(indexes=indexes, window=window, out_dtype='float32')
                    
                    # 补齐到 3 个通道 (C, H, W)
                    # 例如：如果是 1 个波段，复制成 3 份；如果是 2 个波段，复制最后 1 份
                    current_c = window_data.shape[0]
                    if current_c < 3:
                        last_band = window_data[-1:, :, :] # 取最后一个波段 (1, H, W)
                        repeat_times = 3 - current_c
                        # 拼接: 原始数据 + 重复的最后一个波段
                        window_data = np.concatenate([window_data] + [last_band] * repeat_times, axis=0)

                # window_data shape: (C, H, W) -> 转换为 (H, W, C)
                window_data = np.transpose(window_data, (1, 2, 0))
                
                # 4. 数据类型转换与归一化
                max_val = window_data.max()
                if max_val > 1.0:
                    window_data = window_data / max_val
                
                # 5. 转换为 PIL Image
                pil_img = Image.fromarray((window_data * 255).astype(np.uint8))

        except Exception as e:
            print(f"Failed to load {path}: {e}")
            # 返回一个空白图像避免训练中断
            pil_img = Image.new("RGB", (self.crop_size, self.crop_size), (0, 0, 0))

        target = self.get_target(index)
        target = self.target_decoder(target).decode()
        
        if self.transforms is not None:
            pil_img, target = self.transforms(pil_img, target)
        self.iter_count += 1
        if self.iter_count % 10 == 0:
            gc.collect()
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
