"""
Module to define classes used to load values from manifest dataframes
"""

import torch
import numpy as np

from ..image import tiff_loader_CZYX, png_loader

class LoadColumns:
    """
    Loader class, used to retrieve fields directly from dataframe columns
    """
    def __init__(self, columns):
        self.columns = columns

    def __call__(self, row):
        return {column: row[column] for column in self.columns}


class LoadClass:
    """
    Loader class, used to retrieve class values from the dataframe,
    """
    def __init__(self, num_classes, y_encoded_label, binary=False):
        self.num_classes = num_classes
        self.binary = binary
        self.y_encoded_label = y_encoded_label

    def __call__(self, row):
        if self.binary:
            return torch.tensor([row[str(i)] for i in range(self.num_classes)])

        return torch.tensor(row[self.y_encoded_label])


class Load2DImage:
    """
    Loader class, used to retrieve images from paths given in a dataframe column
    """
    def __init__(self, chosen_col, num_channels, channel_indexes, transform):
        self.chosen_col = chosen_col
        self.num_channels = num_channels
        self.channel_indexes = channel_indexes
        self.transform = transform

    def __call__(self, row):
        return png_loader(
            row[self.chosen_col],
            channel_order="CYX",
            indexes={"C": self.channel_indexes or range(self.num_channels)},
            transform=self.transform,
        )


class Load3DImage:
    """
    Loader class, used to retrieve images from paths given in a dataframe column
    """
    def __init__(self, chosen_col, num_channels, select_channels, transform=None):
        self.chosen_col = chosen_col
        self.num_channels = num_channels
        self.select_channels = select_channels
        self.transform = transform

    def __call__(self, row):
        return tiff_loader_CZYX(
            path_str=row[self.chosen_col],
            select_channels=self.select_channels,
            output_dtype=np.float32,
            channel_masks=None,
            mask_thresh=0,
            transform=self.transform,
        )
