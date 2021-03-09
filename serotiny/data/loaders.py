"""
Module to define classes used to load values from manifest dataframes
"""

import torch
import numpy as np

from ..image import tiff_loader_CZYX, png_loader
from serotiny.models.utils import index_to_onehot


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


class LoadOneHotClass:
    """
    Loader class, used to retrieve class values from the dataframe,
    """

    def __init__(self, num_classes, y_encoded_label):
        self.num_classes = num_classes
        self.y_encoded_label = y_encoded_label

    def __call__(self, row):
        x_cond = torch.tensor(row[self.y_encoded_label])
        x_cond = torch.unsqueeze(x_cond, 0)
        x_cond = torch.unsqueeze(x_cond, 0)
        x_cond_one_hot = index_to_onehot(x_cond, self.num_classes)
        x_cond_one_hot = x_cond_one_hot.squeeze(0)
        return x_cond_one_hot


class LoadSpharmCoeffs:
    """
    Loader class, used to retrieve spharm coeffs from the dataframe,
    """

    def __init__(self):
        pass

    def __call__(self, row):
        dna_spharm_cols = [col for col in row.keys() if "dna_shcoeffs" in col]
        # mem_spharm_cols = [col for col in row.keys() if "mem_shcoeffs" in col]
        # str_spharm_cols = [col for col in row.keys() if "str_shcoeffs" in col]
        return torch.tensor(row[dna_spharm_cols])


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
