from typing import Union, Dict, List
import numpy as np
from typing import Sequence, Tuple
from ome_zarr.reader import Reader
from ome_zarr.io import parse_url
from upath import UPath as Path
from monai.data import ImageReader
from monai.utils import ensure_tuple, require_pkg
from monai.config import PathLike
from monai.data.image_reader import _stack_images

import logging

logging.getLogger("ome_zarr").setLevel(logging.WARNING)
logging.getLogger("ome_zarr.reader").setLevel(logging.WARNING)


@require_pkg(pkg_name="upath")
@require_pkg(pkg_name="ome_zarr")
class OmeZarrReader(ImageReader):
    def __init__(self, level=0, image_name="default", channels=None):
        super().__init__()
        self.level = level
        self.image_name = image_name
        self.channels = channels

    def read(self, data: Union[Sequence[PathLike], PathLike]):
        filenames: Sequence[PathLike] = ensure_tuple(data)
        img_ = []
        for path in filenames:
            if self.image_name:
                path = str(Path(path) / self.image_name)

            reader = Reader(parse_url(path))
            node = next(iter(reader()))
            img_.append(node)

        return img_ if len(filenames) > 1 else img_[0]

    def get_data(self, img) -> Tuple[np.ndarray, Dict]:
        img_array: List[np.ndarray] = []

        for img_obj in ensure_tuple(img):
            data = img_obj.data[self.level].compute()[0]
            if self.channels:
                _metadata_channels = img_obj.metadata["name"]
                _channels = [
                    ch if isinstance(ch, int) else _metadata_channels.index(ch)
                    for ch in self.channels
                ]
                data = data[_channels]

            img_array.append(data)

        return _stack_images(img_array, {}), {}

    def verify_suffix(self, filename: Union[Sequence[PathLike], PathLike]) -> bool:
        for fname in ensure_tuple(filename):
            if not str(fname).endswith("zarr"):
                return False
        return True
