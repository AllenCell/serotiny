"""
Module to define classes used to load values from manifest dataframes
"""

import torch
import numpy as np

from .image import tiff_loader, png_loader

__all__ = [
    "LoadColumns",
    "LoadClass",
    "Load2DImage",
    "Load3DImage"
]

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
    A hacky one hot loader
    Load one hot encoding, also set it to 0 if
    we dont want any condition
    """

    def __init__(self, num_classes, y_encoded_label, set_zero=False):
        self.num_classes = num_classes
        self.y_encoded_label = y_encoded_label
        self.set_zero = set_zero

    def __call__(self, row):
        x_cond = torch.tensor(row[self.y_encoded_label])
        x_cond = torch.unsqueeze(x_cond, 0)
        x_cond = torch.unsqueeze(x_cond, 0)
        x_cond_one_hot = index_to_onehot(x_cond, self.num_classes)
        x_cond_one_hot = x_cond_one_hot.squeeze(0)
        if self.set_zero:
            x_cond_one_hot[x_cond_one_hot != 0] = 0
        return x_cond_one_hot


class LoadIntClass:
    """
    A hacky int class loader
    Load int values for classes, also set to 0 if
    we dont want any condition
    """

    def __init__(self, num_classes, y_encoded_label, set_zero=False):
        self.num_classes = num_classes
        self.y_encoded_label = y_encoded_label
        self.set_zero = set_zero

    def __call__(self, row):
        x_cond = torch.tensor(row[self.y_encoded_label])
        x_cond = torch.unsqueeze(x_cond, 0)
        x_cond = torch.unsqueeze(x_cond, 0)
        x_cond_one_hot = index_to_onehot(x_cond, self.num_classes)
        # x_cond_one_hot = x_cond_one_hot.squeeze(0)
        x_cond_argmax = torch.argmax(x_cond_one_hot, axis=1)
        if self.set_zero:
            x_cond_argmax[x_cond_argmax != 0] = 0
        x_cond_argmax = x_cond_argmax.squeeze(0)
        return x_cond_argmax


class LoadPCA:
    """
    Loader class, used to retrieve PCA from the dataframe,
    also set to 0 if we dont want any pca values
    """

    def __init__(self, x_label, set_zero=False):
        self.x_label = x_label  # DNA_PC1...
        self.set_zero = set_zero

    def __call__(self, row):
        pca_cols = [
            col for col in row.keys() if self.x_label in col and len(col) == len(self.x_label) + 1
        ]
        pca = torch.tensor(row[pca_cols]).float()
        if self.set_zero:
            pca = torch.zeros(1).squeeze(0)
        return pca


class LoadSpharmCoeffs:
    """
    Loader class, used to retrieve spharm coeffs from the dataframe,
    """

    def __init__(self, x_label):
        self.x_label = x_label

    def __call__(self, row):
        spharm_cols = [col for col in row.keys() if self.x_label in col]
        return torch.tensor(row[spharm_cols])


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

    def __init__(self, chosen_col, select_channels=None, transform=None):
        self.chosen_col = chosen_col
        self.select_channels = select_channels
        self.transform = transform

    def __call__(self, row):
        return tiff_loader(
            path_str=row[self.chosen_col],
            select_channels=self.select_channels,
            output_dtype=np.float32,
            channel_masks=None,
            mask_thresh=0,
            transform=self.transform,
        )

def infer_extension_loader(extension, chosen_col="true_paths"):
    if extension == ".png":
        return Load2DImage(
            chosen_col=chosen_col,
            num_channels=3,
            channel_indexes=[0,1,2],
            transform=None
        )

    if extension == ".tiff":
        return Load3DImage(
            chosen_col=chosen_col,
        )

    raise NotImplementedError(f"Can't determine appropriate loader for given extension {extension}")
