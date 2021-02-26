from collections.abc import Iterable
import warnings
import multiprocessing as mp
from itertools import chain, combinations

import numpy as np
import pandas as pd
import quilt3

from sklearn.preprocessing import OneHotEncoder
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.data.dataloader import default_collate as collate
from serotiny.library.image import png_loader, tiff_loader_CZYX


def powerset(iterable):
    """
    Generate all combinations of the elements of `iterable`, for all possible
    size of the combo
    """
    elements = list(iterable)
    return list(chain.from_iterable(combinations(elements, r)
                for r in range(len(elements) + 1)))

def download_quilt_data(
    package="rorydm/mitotic_annotations",
    bucket="s3://allencell-internal-quilt",
    data_save_loc="quilt_data",
    ignore_warnings=True,
):
    """
    download a quilt dataset and supress nfs file attribe warnings
    """
    dataset_manifest = quilt3.Package.browse(package, bucket)

    if ignore_warnings:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            dataset_manifest.fetch(data_save_loc)
    else:
        dataset_manifest.fetch(data_save_loc)


def sample_classes(manifest, column, classes):
    """
    Sample one of each class from the manifest.
    """

    sample = {key: manifest[manifest[column].isin([key])] for key in classes}

    return sample


class DataframeDataset(Dataset):
    """
    Class to wrap a pandas DataFrame in a pytorch Dataset. In practice, at AICS
    we use this to wrap manifest dataframes that point to the image files that
    correspond to a cell. The `loaders` dict contains a loading function for each
    key, normally consisting of a function to load the contents of a file from a path.
    Parameters
    ----------
    dataframe: pd.DataFrame
        The file which points to or contains the data to be loaded
    loaders: dict
        A dict which contains methods to appropriately load data from columns
        in the dataset.
    """

    def __init__(self, dataframe, loaders=None):
        self.dataframe = dataframe
        self.loaders = loaders

    def __len__(self):
        return len(self.dataframe)

    def _get_single_item(self, idx):
        row = self.dataframe.loc[idx, :]
        return {key: loader(row) for key, loader in self.loaders.items()}

    def __getitem__(self, idx):
        # TODO: look at handling the index as strings
        sample = (
            collate([self._get_single_item(i) for i in idx])
            if (isinstance(idx, Iterable) and not isinstance(idx, str))
            else self._get_single_item(idx)
        )
        return sample

class LoadField:
    """
    Loader class, used to retrieve fields directly from the dataframe
    """
    def __init__(self, fields):
        self.id_fields = fields

    def __call__(self, row):
        return {key: row[field] for field in self.fields}


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


def load_data_loader(
    dataset, loaders, transform, batch_size=16, num_workers=0, shuffle=False
):
    """ Load a pytorch DataLoader from the provided dataset. """

    # Load a dataframe from the dataset, using the provided row processing fns
    dataframe = DataframeDataset(dataset, loaders=loaders, transform=transform)

    # Configure WeightedRandomSampler to handle class imbalances
    weights = [e for e in dataframe.dataframe["ClassWeights"]]
    sampler = WeightedRandomSampler(weights, len(dataframe.dataframe))

    # create the pytorch dataloader from that dataframe
    dataloader = DataLoader(
        dataframe,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=True,
        drop_last=True,
        num_workers=num_workers,
        multiprocessing_context=mp.get_context("fork"),
    )

    return dataloader


def one_hot_encoding(dataset: pd.DataFrame, column: str):
    # Make a one hot encoding for this column in the dataset.

    enc = OneHotEncoder(sparse=False, dtype=np.float64)
    enc.fit(dataset[[column]])
    one_hot = enc.transform(dataset[[column]])

    return one_hot


def append_one_hot(dataset: pd.DataFrame, column: str, id: str):
    # Modifies its argument by appending the one hot encoding columns
    # into the given dataset.

    one_hot = one_hot_encoding(dataset, column)
    # Lets merge on a unique ID to avoid errors here
    dataset = pd.merge(dataset, pd.DataFrame(one_hot, index=dataset[id]), on=[id])

    # Lets also calculate class weights
    labels_unique, counts = np.unique(dataset[column], return_counts=True)
    class_weights = [sum(counts) / c for c in counts]
    class_weights_dict = {i: j for (i, j) in zip(labels_unique, class_weights)}
    weights = [class_weights_dict[e] for e in dataset[column]]
    dataset["ClassWeights"] = weights

    # Convert class labels to integers for NLLLoss
    class_labels_dict = {
        # i: j for (i, j) in zip(labels_unique, range(len(labels_unique)))
        label: index
        for index, label in enumerate(labels_unique)
    }

    dataset[column + "Integer"] = [class_labels_dict[e] for e in dataset[column]]

    return dataset, one_hot.shape[-1]
