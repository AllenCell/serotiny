import multiprocessing as mp
from typing import Sequence, Union

import re
import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, WeightedRandomSampler
from .dataframe_dataset import DataframeDataset

from pathlib import Path

from actk.utils import dataset_utils


def load_csv(dataset: Union[str, Path, pd.DataFrame], required_fields: Sequence[str]):
    """
    Read dataframe from either a path or an existing pd.DataFrame, checking
    the fields given by `required` are present
    """

    # Handle dataset provided as string or path
    if isinstance(dataset, (str, Path)):
        dataset = Path(dataset).expanduser().resolve(strict=True)

        # Read dataset
        dataset = pd.read_csv(dataset)

    # Check that all columns provided as required are in the dataset
    missing_fields = set(required_fields) - set(dataset.columns)
    if len(missing_fields) > 0:
        raise ValueError(f"Some or all of the required fields were not
                           found on the given dataframe:\n{missing_fields}")

    return dataset


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


def filter_columns(cols_to_filter, regex=None, startswith=None, endswith=None,
                   contains=None, excludes=None):
    if regex is not None:
        return [col for col in cols_to_filter if re.match(regex, col)]

    keep = [True] * len(cols_to_filter)
    for i in range(len(cols_to_filter)):
        if startswith is not None:
            keep[i] &= str(cols_to_filter[i]).startswith(startswith)
        if endswith is not None:
            keep[i] &= str(cols_to_filter[i]).endswith(endswith)
        if contains is not None:
            keep[i] &= (contains in str(cols_to_filter[i]))
        if excludes is not None:
            keep[i] &= (excludes not in str(cols_to_filter[i]))

    return [
        col for col, keep_col in zip(cols_to_filter, keep)
        if keep_col
    ]
