"""
Module to define classes used to load values from manifest dataframes
"""

import re
import torch
import numpy as np

from collections import defaultdict
from torchvision import transforms

from serotiny.io.image import tiff_loader_CZYX, png_loader
from serotiny.io.dataframe.utils import filter_columns
from serotiny.utils import get_classes_from_config

__all__ = ["LoadColumns", "LoadClass", "Load2DImage", "Load3DImage"]

class Loader:
    def __init__(self):
        self.set_mode("train")

    def set_mode(self, mode):
        self.mode = mode


class LoadColumn(Loader):
    """
    Loader class, used to retrieve fields directly from dataframe columns
    """

    def __init__(
            self,
            column='index',
            dtype="float"):
        super().__init__()

        self.column = column
        self.dtype = dtype

    def __call__(self, row):
        return row[self.column].astype(self.dtype)


class LoadColumns(Loader):
    """
    Loader class, used to retrieve fields directly from dataframe columns
    """

    def __init__(self, columns=None, startswith=None, endswith=None,
                 contains=None, excludes=None, regex=None, dtype="float"):
        super().__init__()
        self.columns = columns
        self.startswith = startswith
        self.endswith = endswith
        self.contains = contains
        self.excludes = excludes
        self.regex = regex
        self.dtype = dtype

        if columns is None:
            assert (
                (startswith is not None) or
                (endswith is not None) or
                (contains is not None) or
                (excludes is not None) or
                (regex is not None)
            )
        else:
            self.columns = set(columns)

    def _filter_columns(self, cols_to_filter):
        if self.columns is None:
            self.columns = filter_columns(
                cols_to_filter, self.regex, self.startswith, self.endswith,
                self.contains, self.excludes)

        return self.columns


    def __call__(self, row):
        filtered_cols = self._filter_columns(row.index)
        return row[filtered_cols].values.astype(self.dtype)


class LoadClass(Loader):
    """
    Loader class, used to retrieve class values from the dataframe,
    """

    def __init__(self, num_classes, y_encoded_label, binary=False):
        super().__init__()
        self.num_classes = num_classes
        self.binary = binary
        self.y_encoded_label = y_encoded_label

    def __call__(self, row):
        if self.binary:
            return torch.tensor([row[str(i)] for i in range(self.num_classes)])

        return torch.tensor(row[self.y_encoded_label])


class Load2DImage(Loader):
    """
    Loader class, used to retrieve images from paths given in a dataframe column
    """

    def __init__(
            self,
            column='image',
            num_channels=1,
            channel_indexes=None,
            transforms_dict=None):

        super().__init__()
        self.column = column
        self.num_channels = num_channels
        self.channel_indexes = channel_indexes

        self.transforms = defaultdict(None)
        for key, transforms_config in transforms_dict.items():
            self.transforms[key] = load_transforms(transforms_config)


    def __call__(self, row):
        return png_loader(
            row[self.column],
            channel_order="CYX",
            indexes={"C": self.channel_indexes or range(self.num_channels)},
            transform=self.transforms.get(self.mode)
        )


class Load3DImage(Loader):
    """
    Loader class, used to retrieve images from paths given in a dataframe column
    """

    def __init__(
            self,
            column='image',
            select_channels=None,
            transforms_dict=None):
        super().__init__()
        self.column = column
        self.select_channels = select_channels
        transforms_dict = transforms_dict or {}

        self.transforms = defaultdict(None)
        for key, transforms_config in transforms_dict.items():
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


def load_transforms(transforms_dict):
    if transforms_dict is not None:
        return transforms.Compose(
            get_classes_from_config(transforms_dict)
        )
    return None


def infer_extension_loader(extension, column="true_paths"):
    if extension == ".png":
        return Load2DImage(
            column=column,
            num_channels=3,
            channel_indexes=[0, 1, 2],
            transform=None,
        )

    if extension == ".tiff":
        return Load3DImage(
            column=column,
        )

    raise NotImplementedError(
        f"Can't determine appropriate loader for given extension {extension}"
    )
