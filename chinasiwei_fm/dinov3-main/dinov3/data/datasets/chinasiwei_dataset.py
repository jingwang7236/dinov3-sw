import os
from typing import Callable, Optional, List, Any
from .decoders import Decoder, MultiBandTiffDecoder, ImageDataDecoder, TargetDecoder, ChannelSelectTIFFDecoder
from .extended import ExtendedVisionDataset
from typing import Any, Tuple
import numpy as np

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
    ) -> None:
        super().__init__(root=root, transforms=transforms, transform=transform,
                         target_transform=target_transform, image_decoder=image_decoder)
        self.items: List[str] = []
        with open(os.path.join(root, list_file), "r") as f:
            for ln in f:
                p = ln.strip()
                if not p:
                    continue
                self.items.append(p)
                # self.items.append(p if os.path.isabs(p) else os.path.join("images", p))

    def get_image_data(self, index: int) -> bytes:
        full = os.path.join(self.root, self.items[index])
        with open(full, "rb") as f:
            return f.read()

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
