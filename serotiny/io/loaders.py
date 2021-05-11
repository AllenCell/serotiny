"""
Module to define classes used to load values from manifest dataframes
"""

import re
import torch
import numpy as np

from torchvision import transforms

from serotiny.image import tiff_loader_CZYX, png_loader
from serotiny.utils import get_classes_from_config

__all__ = ["LoadColumns", "LoadClass", "Load2DImage", "Load3DImage"]

class Loader:
    def __init__(self):
        self.train()

    def train(self):
        self.mode = "train"

    def eval(self):
        self.mode = "eval"


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

    def _filter(self, cols_to_filter):
        if self.columns is not None:
            return (col for col in cols_to_filter if col in self.columns)

        if self.regex is not None:
            # cache regex results
            self.columns = {
                col for col in cols_to_filter if re.match(self.regex, col)
            }
            return (col for col in cols_to_filter if col in self.columns)

        keep = [True] * len(cols_to_filter)
        for i in range(len(cols_to_filter)):
            if self.startswith is not None:
                keep[i] &= str(cols_to_filter[i]).startswith(self.startswith)
            if self.endswith is not None:
                keep[i] &= str(cols_to_filter[i]).endswith(self.endswith)
            if self.contains is not None:
                keep[i] &= (self.contains in str(cols_to_filter[i]))
            if self.excludes is not None:
                keep[i] &= (self.excludes not in str(cols_to_filter[i]))

        self.columns = {
            col for col, keep_col in zip(cols_to_filter, keep)
            if keep_col
        }
        return (col for col in cols_to_filter if col in self.columns)

    def __call__(self, row):
        return row[[column for column
                    in self._filter(row.index)]].values.astype(self.dtype)


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

    def __init__(self, chosen_col, num_channels, channel_indexes,
                 train_transforms, eval_transforms):
        super().__init__()
        self.chosen_col = chosen_col
        self.num_channels = num_channels
        self.channel_indexes = channel_indexes
        self.train_transform = load_transforms(train_transforms)
        self.eval_transform = load_transforms(eval_transforms)


    def __call__(self, row):
        return png_loader(
            row[self.chosen_col],
            channel_order="CYX",
            indexes={"C": self.channel_indexes or range(self.num_channels)},
            transform=(self.train_transform if self.mode == "train"
                       else self.eval_transform)
        )


class Load3DImage(Loader):
    """
    Loader class, used to retrieve images from paths given in a dataframe column
    """

    def __init__(self, chosen_col, select_channels=None, train_transforms=None,
                 eval_transforms=None):
        super().__init__()
        self.chosen_col = chosen_col
        self.select_channels = select_channels
        self.train_transform = load_transforms(train_transforms)
        self.eval_transform = load_transforms(eval_transforms)

    def __call__(self, row):
        return tiff_loader_CZYX(
            row[self.chosen_col],
            select_channels=self.select_channels,
            output_dtype=np.float32,
            channel_masks=None,
            mask_thresh=0,
            transform=(self.train_transform if self.mode == "train"
                       else self.eval_transform)
        )


def load_transforms(transforms_dict):
    if transforms_dict is not None:
        return transforms.Compose(
            get_classes_from_config(transforms_dict)
        )
    return None


def infer_extension_loader(extension, chosen_col="true_paths"):
    if extension == ".png":
        return Load2DImage(
            chosen_col=chosen_col,
            num_channels=3,
            channel_indexes=[0, 1, 2],
            transform=None,
        )

    if extension == ".tiff":
        return Load3DImage(
            chosen_col=chosen_col,
        )

    raise NotImplementedError(
        f"Can't determine appropriate loader for given extension {extension}"
    )
