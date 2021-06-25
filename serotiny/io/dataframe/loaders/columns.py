from serotiny.io.dataframe.utils import filter_columns
from .abstract_loader import Loader


class LoadColumn(Loader):
    """
    Loader class, used to retrieve fields directly from dataframe columns
    """

    def __init__(
            self,
            column='index',
            dtype="float"):
        super().__init__()

        self.column = column
        self.dtype = dtype

    def __call__(self, row):
        return row[self.column].astype(self.dtype)


class LoadColumns(Loader):
    """
    Loader class, used to retrieve fields directly from dataframe columns
    """

    def __init__(self, columns=None, startswith=None, endswith=None,
                 contains=None, excludes=None, regex=None, dtype="float"):
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
                (startswith is not None) or
                (endswith is not None) or
                (contains is not None) or
                (excludes is not None) or
                (regex is not None)
            )
        else:
            self.columns = set(columns)

    def _filter_columns(self, cols_to_filter):
        if self.columns is None:
            self.columns = filter_columns(
                cols_to_filter, self.regex, self.startswith, self.endswith,
                self.contains, self.excludes)

        return self.columns


    def __call__(self, row):
        filtered_cols = self._filter_columns(row.index)
        return row[filtered_cols].values.astype(self.dtype)
