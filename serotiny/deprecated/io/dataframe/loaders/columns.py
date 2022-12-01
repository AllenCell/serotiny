from typing import Optional, Sequence
import numpy as np

from serotiny.transforms.dataframe.transforms import _filter_columns as filter_columns

from .abstract_loader import Loader


class LoadColumn(Loader):
    """Loader class, used to retrieve fields directly from dataframe
    columns."""

    def __init__(self, column: str, dtype: str = "float", unsqueeze: bool = True):
        """
        Parameters
        ----------
        column: str
            The column to retrieve

        """
        super().__init__()

        self.column = [column] if unsqueeze else column
        self.dtype = np.dtype(dtype).type
        self.unsqueeze = unsqueeze

    def __call__(self, row):
        return self.dtype(row[self.column])


class LoadColumns(Loader):
    """Loader class, used to retrieve fields directly from multiple dataframe
    columns, concatenating them into an array.

    It leverages `filter_columns` to enable using simple queries to select the
    columns to use.
    """

    def __init__(
        self,
        columns: Optional[Sequence[str]] = None,
        startswith: Optional[str] = None,
        endswith: Optional[str] = None,
        contains: Optional[str] = None,
        excludes: Optional[str] = None,
        regex: Optional[str] = None,
        dtype: str = "float",
    ):
        """
        Parameters
        ----------
        columns: Sequence[str]
            Explicit list of columns to include. If it is supplied,
            the remaining filters are ignored

        regex: Optional[str] = None
            A string containing a regular expression to be matched

        startswith: Sequence[str] = None
            A substring the matching columns must start with

        endswith: Sequence[str] = None
            A substring the matching columns must end with

        contains: Sequence[str] = None
            A substring the matching columns must contain

        excludes: Sequence[str] = None
            A substring the matching columns must not contain

        dtype: str = "float"
            dtype of the resulting array

        """
        super().__init__()
        self.columns = columns
        self.startswith = startswith
        self.endswith = endswith
        self.contains = contains
        self.excludes = excludes
        self.regex = regex
        self.dtype = dtype

        if columns is None:
            assert (
                (startswith is not None)
                or (endswith is not None)
                or (contains is not None)
                or (excludes is not None)
                or (regex is not None)
            )
        else:
            self.columns = list(columns)

    def _filter_columns(self, columns_to_filter):
        if self.columns is None:
            self.columns = filter_columns(
                columns_to_filter,
                self.regex,
                self.startswith,
                self.endswith,
                self.contains,
                self.excludes,
            )

        return self.columns

    def __call__(self, row):
        filtered_cols = self._filter_columns(row.index)
        return row[filtered_cols].astype(self.dtype)
