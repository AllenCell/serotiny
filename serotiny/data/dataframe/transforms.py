from typing import Optional, Sequence
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from serotiny.io.dataframe import filter_columns as _filter_columns


def split_dataframe(
    dataframe: pd.DataFrame,
    train_frac: float,
    val_frac: Optional[float] = None,
    return_splits: bool = True,
):
    """
    Given a pandas dataframe, perform a train-val-test split and either
    return three different dataframes, or append a column identifying the
    split each row belongs to.

    TODO: extend this to enable balanced / stratified splitting

    Parameters
    ----------
    dataframe: pd.DataFrame
        Input dataframe

    train_frac: float
        Fraction of data to use for training. Must be <= 1

    val_frac: Optional[float]
        Fraction of data to use for validation. By default,
        the data not used for training is split in half
        between validation and test

    return_splits: bool = True
        Whether to return the three splits separately, or to append
        a column to the existing dataframe and return the modified
        dataframe
    """

    train_ix, val_test_ix = train_test_split(
        dataframe.index.tolist(), train_size=train_frac
    )
    if val_frac is not None:
        val_frac = val_frac / (1 - train_frac)
    else:
        # by default use same size for val and test
        val_frac = 0.5

    val_ix, test_ix = train_test_split(val_test_ix, train_size=val_frac)

    if return_splits:
        return dict(
            train=dataframe.loc[train_ix],
            val=dataframe.loc[val_ix],
            test=dataframe.loc[test_ix],
        )

    dataframe.loc[train_ix, "split"] = "train"
    dataframe.loc[val_ix, "split"] = "valid"
    dataframe.loc[test_ix, "split"] = "test"

    return dataframe


def filter_rows(dataframe: pd.DataFrame, column: str, values: Sequence, exclude: bool = False):
    """
    Filter a dataframe, keeping only the rows where a given
    column's value is contained in a list of values

    Parameters
    ----------
    dataframe: pd.DataFrame
        Input dataframe

    column: str
        The column to be used for filtering

    values: Sequence
        List of values to filter for
    """

    if exclude:
        return dataframe.loc[~dataframe[column].isin(values)]
    else:
        return dataframe.loc[dataframe[column].isin(values)]


def filter_columns(
    dataframe: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    startswith: Optional[str] = None,
    endswith: Optional[str] = None,
    contains: Optional[str] = None,
    excludes: Optional[str] = None,
    regex: Optional[str] = None,
):
    """
    Select columns in a dataset, using different filtering options.
    See serotiny.io.dataframe.readers.filter_columns for more details.

    Parameters
    ----------
    columns: Sequence[str]
        Explicit list of columns to include. If it is supplied,
        the remaining filters are ignored

    startswith: Sequence[str] = None
        A substring the matching columns must start with

    endswith: Sequence[str] = None
        A substring the matching columns must end with

    contains: Sequence[str] = None
        A substring the matching columns must contain

    excludes: Sequence[str] = None
        A substring the matching columns must not contain

    regex: Optional[str] = None
        A string containing a regular expression to be matched
    """

    if columns is None:
        columns = _filter_columns(
            dataframe.columns.tolist(), regex, startswith, endswith, contains, excludes
        )
    return dataframe[columns]


def sample_n_each(
    dataframe: pd.DataFrame,
    column: str,
    number: int = 1,
    force: bool = False,
    seed: int = 42,
):
    """
    Transform a dataframe to have equal number of rows per value of `column`.

    In case a given value of `column` has less than `number` corresponding rows:
    - if `force` is True the corresponding rows are sampled with replacement
    - if `force` is False all the rows are given for that value

    Parameters
    ----------
    dataframe: pd.DataFrame
        Input dataframe

    column: str
        The column to be used for selection

    number: int
        Number of rows to include per unique value of `column`

    force: bool = False
        Toggle upsampling of classes with number of samples smaller
        than `number`

    seed: int
        Random seed used for sampling
    """

    values = dataframe[column].unique()

    subsets = []
    for value in values:
        class_rows = dataframe[dataframe[column] == value]
        if force or (len(class_rows) >= number):
            subsets.append(
                class_rows.sample(
                    number,
                    random_state=seed,
                    # only sample with replacement if there
                    # aren't enough data points in this class
                    replace=(len(class_rows) < number),
                )
            )
        else:
            subsets.append(class_rows.sample(frac=1, random_state=seed))

    return pd.concat(subsets)


def append_one_hot(dataframe: pd.DataFrame, column: str):
    """
    Modifies its argument by appending the one hot encoding columns
    into the given dataframe. Calls function one_hot_encoding

    Parameters
    -----------
    dataframe: pd.DataFrame
        Input dataframe

    column: str
        Column to convert into one hot encoding

    """

    one_hot = OneHotEncoder(sparse=False).fit_transform(dataframe[[column]])

    for idx in range(one_hot.shape[1]):
        dataframe[f"{column}_one_hot_{idx}"] = one_hot[:, idx]

    return dataframe


def append_labels_to_integers(dataframe: pd.DataFrame, column: str):
    """
    Modifies its argument by appending the integer-encoded values of `column`
    into the given dataframe.

    Parameters
    -----------
    dataframe: pd.DataFrame
        Input dataframe

    column: str
        Column to convert into one hot encoding

    """

    dataframe[f"{column}_integer"] = LabelEncoder().fit_transform(dataframe[[column]])

    return dataframe


def append_class_weights(dataframe: pd.DataFrame, column: str):
    """
    Add class weights (based on `column`) to a dataframe
    """
    labels_unique, counts = np.unique(dataframe[column], return_counts=True)
    class_weights = [sum(counts) / c for c in counts]
    class_weights_dict = dict(zip(labels_unique, class_weights))
    weights = [class_weights_dict[e] for e in dataframe[column]]
    dataframe[f"{column}_class_weights"] = weights
    return dataframe
