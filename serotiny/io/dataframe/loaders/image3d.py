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

    def __init__(
            self,
            column='image',
            select_channels=None,
            transforms=None):
        super().__init__()
        self.column = column
        self.select_channels = select_channels
        transforms = transforms or []

        self.transforms = defaultdict(None)
        for key, transforms_config in transforms.items():
            self.transforms[key] = load_transforms(transforms_config)


    def __call__(self, row):
        return tiff_loader_CZYX(
            row[self.column],
            select_channels=self.select_channels,
            output_dtype=np.float32,
            channel_masks=None,
            mask_thresh=0,
            transform=self.transforms.get(self.mode)
        )
