from collections import defaultdict
import numpy as np

from aicsfiles import FileManagementSystem

from serotiny.io.image import tiff_loader_CZYX
from .abstract_loader import Loader
from .utils import load_transforms

class Load3DImage(Loader):
    """
    Loader class, used to retrieve images from paths given in a dataframe column
    """

    def __init__(self, chosen_col, select_channels=None, transforms_dict={}, fms=False):
        super().__init__()
        self.chosen_col = chosen_col
        self.select_channels = select_channels

        self.transforms = defaultdict(None)
        for key, transforms_config in transforms_dict.items():
            self.transforms[key] = load_transforms(transforms_config)

        self.fms = (FileManagementSystem() if fms else None)

    def _get_path(self, row):
        if self.fms is not None:
            return self.fms.get_file_by_id(row[self.chosen_col]).path
        return row[self.chosen_col]

    def __call__(self, row):
        return tiff_loader_CZYX(
            self._get_path(row),
            select_channels=self.select_channels,
            output_dtype=np.float32,
            channel_masks=None,
            mask_thresh=0,
            transform=self.transforms[self.mode]
        )
