import warnings
import multiprocessing as mp
from itertools import chain, combinations
from typing import Union

import numpy as np
import pandas as pd
import quilt3
import torch

from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, WeightedRandomSampler
from .dataframe_dataset import DataframeDataset


def powerset(iterable):
    """
    Generate all combinations of the elements of `iterable`, for all possible
    size of the combo
    """
    elements = list(iterable)
    return list(
        chain.from_iterable(combinations(elements, r) for r in range(len(elements) + 1))
    )


def download_quilt_data(
    package: str,
    bucket: str,
    data_save_loc: str,
    ignore_warnings=True,
):
    """
    Download a quilt dataset and supress nfs file attribe warnings

    Parameters
    ----------
    package: str
        Name of the package on s3.
        Example: "aics/hipsc_single_cell_image_dataset"

    bucket: str
        The s3 bucket storing the package
        Example: "s3://allencell"

    data_save_loc: str,
        Path to save data

    ignore_warnings: bool,
        Whether to suppress nfs file attribute warnings or not
    """
    dataset_manifest = quilt3.Package.browse(package, bucket)

    if ignore_warnings:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            dataset_manifest.fetch(data_save_loc)
    else:
        dataset_manifest.fetch(data_save_loc)


def load_data_loader(
    dataset: pd.DataFrame,
    loaders: dict,
    transform: Union[list, torch.nn.Module],
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    weights_col="ClassWeights",
):
    """
    Load a pytorch DataLoader from the provided dataset.

    Parameters
    ----------
    dataset: pd.DataFrame
        Dataframe to convert into a pytorch dataloader. Pass in a separate dataframe
        for train, val and test to make dataloaders for each

    loaders: dict
        Dictionary containing information about how to load the CellID, Image
        and Label from the dataframe per row.

    transform: Union[list, torch.nn.Module]
        Pytorch transforms to apply to the images

    batch_size: int
        Batch size used in the dataloader. Example: 64

    num_workers: int
        Number of worker processes to create in the dataloader

    shuffle: bool
        Whether to shuffle the dataset or not. Example: True for train dataloader

    weights_col: str
        Use a column if class weights to instantiate a weighted random sampler
        that handles class imbalances

    Returns
    --------
    dataloader
        A pytorch dataloader based on a pandas dataframe

    """

    # Load a dataframe from the dataset, using the provided row processing fns
    dataframe = DataframeDataset(dataset, loaders=loaders, transform=transform)

    # Configure WeightedRandomSampler to handle class imbalances
    sampler = None
    if weights_col is not None:
        weights = dataframe.dataframe[weights_col].values
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
    """
    Make one hot encoding of a column in a pandas dataframe

    Parameters
    -----------
    dataset: pd.DataFrame
        Input dataframe

    columns: str
        Column to convert into one hot encoding

    Returns
    ---------
    one_hot: np.array
        One hot encoding of input column
    """
    # Make a one hot encoding for this column in the dataset.

    enc = OneHotEncoder(sparse=False, dtype=np.float64)
    enc.fit(dataset[[column]])
    one_hot = enc.transform(dataset[[column]])

    return one_hot


def append_one_hot(dataset: pd.DataFrame, column: str, index: str):
    """
    Modifies its argument by appending the one hot encoding columns
    into the given dataset. Calls function one_hot_encoding

    Parameters
    -----------
    dataset: pd.DataFrame
        Input dataframe

    column: str
        Column to convert into one hot encoding

    index: str
        Index to merge the one hot encoding back onto original dataframe
    """

    one_hot = one_hot_encoding(dataset, column)
    # Lets merge on a unique ID to avoid errors here
    dataset = pd.merge(dataset, pd.DataFrame(one_hot, index=dataset[index]), on=[index])

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
