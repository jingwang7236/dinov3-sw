import os
from typing import Any, Callable, List, Optional, Tuple, Union, Dict
from ..decoders import DecoderType, TargetDecoder
from ..extended import ExtendedVisionDataset
import numpy as np



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
        image_decoder_type: DecoderType = DecoderType.ChannelSelectTIFFDecoder,
        image_decoder_params: Dict[str, Any] = {},
        **kwargs: Any,
    ) -> None:
        super().__init__(
            root,
            transforms,
            transform,
            target_transform,
            image_decoder_type=image_decoder_type,
            image_decoder_params={
                "select_channel": True
            },
            **kwargs,
        )

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
            # print(self._image_paths[index], channels)
            image_data = image_data + bytes(channels) + (len(channels)).to_bytes(1, byteorder="big")

            # L = image_data[-1]
            # _channel = list(image_data[-1 - L: -1])
            # print(f"img_data[-1]: {L}, channel: {_channel}")
            return image_data
        else:
            return image_data


    def get_target(self, index: int) -> Any:
        return None

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        try:
            image_data = self.get_image_data(index)
            image = self._image_decoder_class(image_data, **self._decoder_params).decode()
        except Exception as e:
            print(f"failed to load {self._image_paths[index]}")
            raise RuntimeError(f"can not read image for sample {index}") from e
        target = self.get_target(index)
        target = TargetDecoder(target).decode()

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


    def __len__(self) -> int:
        return len(self._image_paths)


