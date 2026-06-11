# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from io import BytesIO
from typing import Any

from PIL import Image

import numpy as np
try:
    import rasterio as rio
    from rasterio.errors import NotGeoreferencedWarning
    import warnings
    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
except ImportError:
    rio = None

try:
    import tifffile
except ImportError:
    print("Could not import `tifffile`, ChannelSelectTIFFDecoder will be disabled")


class Decoder:
    def decode(self) -> Any:
        raise NotImplementedError


class ImageDataDecoder(Decoder):
    def __init__(self, image_data: bytes) -> None:
        self._image_data = image_data

    def decode(self) -> Image:
        f = BytesIO(self._image_data)
        return Image.open(f).convert(mode="RGB")


class TargetDecoder(Decoder):
    def __init__(self, target: Any):
        self._target = target

    def decode(self) -> Any:
        return self._target


class DenseTargetDecoder(Decoder):
    def __init__(self, image_data: bytes) -> None:
        self._image_data = image_data

    def decode(self) -> Image:
        f = BytesIO(self._image_data)
        return Image.open(f)



class MultiBandTiffDecoder(Decoder):
    """Decode TIFF bytes to HWC float32 numpy array (0..1)."""
    def __init__(self, image_data: bytes) -> None:
        self._image_data = image_data

    def decode(self) -> Image:
        # f = BytesIO(self._image_data)
        assert rio is not None, "Please install rasterio for TIFF decoding"
        with rio.MemoryFile(self._image_data) as mem:
            with mem.open() as src:
                nb = src.count  # 波段数量
                H, W = src.height, src.width
                
                if nb == 3:
                    arr = src.read(indexes=(1,2,3), out_dtype="float32")  # C,H,W
                elif nb < 3:
                    # 读取所有band，并pad到三个
                    idx = list(range(1, nb + 1))
                    arr = src.read(indexes=idx, out_dtype="float32")
                    pad_cnt = 3 - nb
                    last = arr[-1:, ...]
                    arr = np.concatenate([arr, np.repeat(last, pad_cnt, axis=0)], axis=0)
                else:
                    # 多光谱随机选择3个波段
                    choice = np.random.choice(nb, size=3, replace=False) + 1
                    arr = src.read(indexes=choice.tolist(), out_dtype="float32")
        arr = np.moveaxis(arr, 0, -1)  # -> HxWxC
        arr = arr.astype(np.float32)
        # 这里做归一化/标准化（按你的波段范围定制）
        arr /= np.clip(arr.max(), 1.0, None)

        image = (arr * 255.0).round().astype(np.uint8)
        image = np.ascontiguousarray(image)
        image = Image.fromarray(image, mode="RGB")

        return image


class ChannelSelectTIFFDecoder(Decoder):
    def __init__(self, image_data: bytes, select_channel: bool = True) -> None:
        self.select_channel = select_channel
        if select_channel:
            L = image_data[-1]
            self._channel = list(image_data[-1 - L: -1])
            self._image_data = image_data[:-1 - L]

            # self._image_data = image_data[:-1]
            # self._channel = image_data[-1]
        else:
            self._image_data = image_data

    def decode(self):

        assert rio is not None, "Please install rasterio for TIFF decoding"
        with rio.MemoryFile(self._image_data) as mem:
            with mem.open() as src:
                arr = src.read(out_dtype="float32")  # C,H,W
        # if arr.ndim == 2:
        #     arr = arr[None, ...]

        arr = np.moveaxis(arr, 0, -1)  # -> HxWxC
        arr = arr.astype(np.float32)

        arr /= np.clip(arr.max(), 1.0, None)
        image = (arr * 255.0).round().astype(np.uint8)
        image = np.ascontiguousarray(image)
        
        if self.select_channel:
            image = image[:, :, self._channel]
            image = image.squeeze(-1)  # 为适配PIl  转成(H,W)
            image = Image.fromarray(image, mode="L")
            return image
        
        image = Image.fromarray(image, mode="RGB")
        return image

        # if self.select_channel:
        #     return torch.Tensor(image).permute(2, 0, 1)[[self._channel]]
        # return torch.Tensor(image).permute(2, 0, 1)
