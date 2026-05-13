# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from io import BytesIO
from typing import Any, Type

from PIL import Image
import numpy as np
import torch
from enum import Enum

try:
    import tifffile
except ImportError:
    print("Could not import `tifffile`, TIFFImageDataDecoder will be disabled")

try:
    import rasterio as rio
    from rasterio.errors import NotGeoreferencedWarning
    import warnings
    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
except ImportError:
    rio = None

class Decoder:
    def decode(self) -> Any:
        raise NotImplementedError


class DecoderType(Enum):
    ImageDataDecoder = "ImageDataDecoder"
    XChannelsDecoder = "XChannelsDecoder"
    XChannelsTIFFDecoder = "XChannelsTIFFDecoder"
    ChannelSelectDecoder = "ChannelSelectDecoder"
    ChannelSelectTIFFDecoder = "ChannelSelectTIFFDecoder"

    def get_class(self) -> Type[Decoder]:  # noqa: C901
        if self == DecoderType.ImageDataDecoder:
            return ImageDataDecoder
        if self == DecoderType.XChannelsDecoder:
            return XChannelsDecoder
        if self == DecoderType.XChannelsTIFFDecoder:
            return XChannelsTIFFDecoder
        if self == DecoderType.ChannelSelectDecoder:
            return ChannelSelectDecoder
        if self == DecoderType.ChannelSelectTIFFDecoder:
            return ChannelSelectTIFFDecoder


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


class XChannelsDecoder(Decoder):
    def __init__(self, image_data: bytes) -> None:
        self._image_data = image_data

    def decode(self):
        im = np.asarray(Image.open(BytesIO(self._image_data)))
        if len(im.shape) == 2:
            im = np.reshape(im, (im.shape[0], im.shape[0], -1), order="F")
        return torch.Tensor(im).permute(2, 0, 1)


class XChannelsTIFFDecoder(Decoder):
    def __init__(self, image_data: bytes, num_channels: int = 3) -> None:
        self._image_data = image_data
        self._num_channels = num_channels

    def decode(self):
        numpy_array = tifffile.imread(BytesIO(self._image_data))
        numpy_array = np.reshape(numpy_array, (numpy_array.shape[0], -1, self._num_channels), order="F")
        return torch.Tensor(numpy_array).permute(2, 0, 1)


class ChannelSelectDecoder(Decoder):
    def __init__(self, image_data: bytes, select_channel: bool = False) -> None:
        self.select_channel = select_channel
        if select_channel:
            self._image_data = image_data[:-1]
            self._channel = image_data[-1]
        else:
            self._image_data = image_data

    def decode(self):
        im = np.asarray(Image.open(BytesIO(self._image_data)))
        if self.select_channel:
            return torch.Tensor(im).permute(2, 0, 1)[[self._channel]]
        return torch.Tensor(im).permute(2, 0, 1)


class ChannelSelectTIFFDecoder(Decoder):
    def __init__(self, image_data: bytes, select_channel: bool = False) -> None:
        self.select_channel = select_channel
        # import ipdb
        # ipdb.set_trace()
        if select_channel:
            L = image_data[-1]
            self._channel = list(image_data[-1 - L: -1])
            self._image_data = image_data[:-1 - L]

            # self._image_data = image_data[:-1]
            # self._channel = image_data[-1]
        else:
            self._image_data = image_data

    def decode(self):
        # im = tifffile.imread(BytesIO(self._image_data)) # (H, W, channel) / (H, W) 
        # if im.ndim == 2:
        #     im = im[..., None]
        
        # im = im.astype(np.float32)


        assert rio is not None, "Please install rasterio for TIFF decoding"
        with rio.MemoryFile(self._image_data) as mem:
            with mem.open() as src:
                arr = src.read(out_dtype="float32")  # C,H,W
        # if arr.ndim == 2:
        #     arr = arr[None, ...]

        arr = np.moveaxis(arr, 0, -1)  # -> HxWxC
        im = arr.astype(np.float32)

        # 这里做归一化/标准化（按你的波段范围定制）
        im /= np.clip(im.max(), 1.0, None)
        image = (im * 255.0).round().astype(np.uint8)
        image = np.ascontiguousarray(image)
        if self.select_channel:
            return torch.Tensor(image).permute(2, 0, 1)[[self._channel]]
        return torch.Tensor(image).permute(2, 0, 1)
