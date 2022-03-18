import re
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd


def split_dataframe(
    dataframe: pd.DataFrame,
    train_frac: float,
    val_frac: Optional[float] = None,
    return_splits: bool = True,
):
    """Given a pandas dataframe, perform a train-val-test split and either return three
    different dataframes, or append a column identifying the split each row belongs to.

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

    # import here to optimize CLIs / Fire usage
    from sklearn.model_selection import train_test_split

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
            valid=dataframe.loc[val_ix],
            test=dataframe.loc[test_ix],
        )

    dataframe.loc[train_ix, "split"] = "train"
    dataframe.loc[val_ix, "split"] = "valid"
    dataframe.loc[test_ix, "split"] = "test"

    return dataframe


def filter_rows(
    dataframe: pd.DataFrame, column: str, values: Sequence, exclude: bool = False
):
    """Filter a dataframe, keeping only the rows where a given column's value is
    contained in a list of values.

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


def _filter_columns(
    columns_to_filter: Sequence[str],
    regex: Optional[str] = None,
    startswith: Optional[str] = None,
    endswith: Optional[str] = None,
    contains: Optional[str] = None,
    excludes: Optional[str] = None,
) -> Sequence[str]:
    """Filter a list of columns, using a combination of different queries, or a `regex`
    pattern. If `regex` is supplied it takes precedence and the remaining arguments are
    ignored. Otherwise, the logical AND of the supplied filters is applied, i.e. the
    columns that respect all of the supplied conditions are returned.

    Parameters
    ----------
    columns_to_filter: Sequence[str]
        List of columns to filter

    regex: Optional[str] = None
        A string containing a regular expression to be matched

    startswith: Optional[str] = None
        A substring the matching columns must start with

    endswith: Optional[str] = None
        A substring the matching columns must end with

    contains: Optional[str] = None
        A substring the matching columns must contain

    excludes: Optional[str] = None
        A substring the matching columns must not contain
    """
    if regex is not None:
        return [col for col in columns_to_filter if re.match(regex, col)]

    keep = [True] * len(columns_to_filter)
    for i in range(len(columns_to_filter)):
        if startswith is not None:
            keep[i] &= str(columns_to_filter[i]).startswith(startswith)
        if endswith is not None:
            keep[i] &= str(columns_to_filter[i]).endswith(endswith)
        if contains is not None:
            keep[i] &= contains in str(columns_to_filter[i])
        if excludes is not None:
            keep[i] &= excludes not in str(columns_to_filter[i])

    return [col for col, keep_column in zip(columns_to_filter, keep) if keep_column]


def filter_columns(
    input: Union[pd.DataFrame, Sequence[str]],
    columns: Optional[Sequence[str]] = None,
    startswith: Optional[str] = None,
    endswith: Optional[str] = None,
    contains: Optional[str] = None,
    excludes: Optional[str] = None,
    regex: Optional[str] = None,
):
    """Select columns in a dataset, using different filtering options. See
    serotiny.data.dataframe.transforms.filter_columns for more details.

    Parameters
    ----------
    input: Union[pd.DataFrame, Sequence[str]]
        The input to operate on. It can either be a pandas DataFrame,
        in which case the result is a DataFrame with only the columns
        that match the filters; or it can be a list of strings, and
        in that case the result is a list containing only the strings
        that match the filters

    columns: Optional[Sequence[str]] = None
        Explicit list of columns to include. If it is supplied,
        the remaining filters are ignored

    startswith: Optional[str] = None
        A substring the matching columns must start with

    endswith: Optional[str] = None
        A substring the matching columns must end with

    contains: Optional[str] = None
        A substring the matching columns must contain

    excludes: Optional[str] = None
        A substring the matching columns must not contain

    regex: Optional[str] = None
        A string containing a regular expression to be matched
    """

    if isinstance(input, pd.DataFrame):
        if columns is None:
            columns = _filter_columns(
                input.columns.tolist(), regex, startswith, endswith, contains, excludes
            )
            return input[columns]
    return _filter_columns(
        input.columns.tolist(), regex, startswith, endswith, contains, excludes
    )


def sample_n_each(
    dataframe: pd.DataFrame,
    column: str,
    number: int = 1,
    force: bool = False,
    seed: int = 42,
):
    """Transform a dataframe to have equal number of rows per value of `column`.

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
    """Modifies its argument by appending the one hot encoding columns into the given
    dataframe. Calls function one_hot_encoding.

    Parameters
    -----------
    dataframe: pd.DataFrame
        Input dataframe

    column: str
        Column to convert into one hot encoding
    """

    # import here to optimize CLIs / Fire usage
    from sklearn.preprocessing import OneHotEncoder

    one_hot = OneHotEncoder(sparse=False).fit_transform(dataframe[[column]])

    for idx in range(one_hot.shape[1]):
        dataframe[f"{column}_one_hot_{idx}"] = one_hot[:, idx]

    return dataframe


def append_labels_to_integers(dataframe: pd.DataFrame, column: str):
    """Modifies its argument by appending the integer-encoded values of `column` into
    the given dataframe.

    Parameters
    -----------
    dataframe: pd.DataFrame
        Input dataframe

    column: str
        Column to convert into one hot encoding
    """

    # import here to optimize CLIs / Fire usage
    from sklearn.preprocessing import LabelEncoder

    dataframe[f"{column}_integer"] = LabelEncoder().fit_transform(dataframe[[column]])

    return dataframe


def append_class_weights(dataframe: pd.DataFrame, column: str):
    """Add class weights (based on `column`) to a dataframe.

    Parameters
    -----------
    dataframe: pd.DataFrame
        Input dataframe

    column: str
        Column to base the weights on
    """
    labels_unique, counts = np.unique(dataframe[column], return_counts=True)
    class_weights = [sum(counts) / c for c in counts]
    class_weights_dict = dict(zip(labels_unique, class_weights))
    weights = [class_weights_dict[e] for e in dataframe[column]]
    dataframe[f"{column}_class_weights"] = weights
    return dataframe


def make_random_df(columns: Sequence[str] = list("ABCD"), n_rows: int = 100):
    """Generate a random dataframe. Useful to test data wrangling pipelines.

    Parameters
    ----------
    columns: Sequence[str] = ["A","B","C","D"]
        List of columns to add to the random dataframe. If none are provided,
        a dataframe with columns ["A","B","C","D"] is created

    n_rows: int = 100
        Number of rows to create for the random dataframe
    """

    data = np.random.randn(n_rows, len(columns))
    return pd.DataFrame(data, columns=columns)
