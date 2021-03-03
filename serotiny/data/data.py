import warnings
import multiprocessing as mp
from itertools import chain, combinations

import numpy as np
import pandas as pd
import quilt3

from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, WeightedRandomSampler
from ..image import png_loader, tiff_loader_CZYX
from .dataframe_dataset import DataframeDataset


def powerset(iterable):
    """
    Generate all combinations of the elements of `iterable`, for all possible
    size of the combo
    """
    elements = list(iterable)
    return list(chain.from_iterable(combinations(elements, r)
                for r in range(len(elements) + 1)))

def download_quilt_data(
    package,
    bucket,
    data_save_loc,
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

def load_data_loader(
        dataset, loaders, transform, batch_size=16, num_workers=0, shuffle=False,
        weights_col="ClassWeights"
):
    """ Load a pytorch DataLoader from the provided dataset. """

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
    # Make a one hot encoding for this column in the dataset.

    enc = OneHotEncoder(sparse=False, dtype=np.float64)
    enc.fit(dataset[[column]])
    one_hot = enc.transform(dataset[[column]])

    return one_hot


def append_one_hot(dataset: pd.DataFrame, column: str, index: str):
    """
    Modifies its argument by appending the one hot encoding columns
    into the given dataset.
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
