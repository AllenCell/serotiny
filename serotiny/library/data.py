from collections.abc import Iterable
import numpy as np
import pandas as pd
import quilt3
import warnings

import multiprocessing as mp

from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.data.dataloader import default_collate as collate


def download_quilt_data(
    package="rorydm/mitotic_annotations",
    bucket="s3://allencell-internal-quilt",
    data_save_loc="quilt_data",
    ignore_warnings=True,
):
    """download a quilt dataset and supress nfs file attribe warnings by default"""
    dataset_manifest = quilt3.Package.browse(package, bucket)

    if ignore_warnings:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            dataset_manifest.fetch(data_save_loc)
    else:
        dataset_manifest.fetch(data_save_loc)


class DataframeDataset(Dataset):
    """general dataframe Dataset class"""

    def __init__(self, dataframe, loaders=None, transform=None):
        self.dataframe = dataframe
        self.loaders = loaders
        self.transform = transform

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


def load_data_loader(
    dataset, loaders, transform, batch_size=16, num_workers=0, shuffle=False
):
    """ Load a pytorch DataLoader from the provided dataset. """

    # Load a dataframe from the dataset, using the provided row processing functions
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
    # TODO make ChosenMitoticClass not hardcoded
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
