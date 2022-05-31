import numpy as np
from torch.utils.data import Dataset, default_collate


class _Row:
    """Helper class to enable string indexing of numpy arrays."""

    def __init__(self, array, columns):
        self.array = array
        self.columns = columns
        self.index = self.columns

    def __getitem__(self, col):
        if isinstance(col, (list, tuple)):
            return self.array[[self.columns.index(_col) for _col in col]]
        return self.array[self.columns.index(col)]

    def __getattr__(self, col):
        return self.array[self.columns.index(col)]


class DataframeDataset(Dataset):
    """Class to wrap a pandas DataFrame in a pytorch Dataset. In practice, at
    AICS we use this to wrap manifest dataframes that point to the image files
    that correspond to a cell. The `loaders` dict contains a loading function
    for each key, normally consisting of a function to load the contents of a
    file from a path.

    Parameters
    ----------
    dataframe: pd.DataFrame
        The file which points to or contains the data to be loaded

    loaders: dict
        A dict which contains methods to appropriately load data from columns
        in the dataset.
    """

    def __init__(self, dataframe, loaders):

        # store only the numpy array containing all the values. this is because
        # in multiprocessing settings, having a pandas dataframe here can incur
        # in excessive memory use, because of copy-on-write memory sharing mechanisms,
        # triggered by pandas' inner workings
        self.dataframe = dataframe.values
        self.columns = dataframe.columns.tolist()

        self.loaders = loaders

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = _Row(self.dataframe[idx], self.columns)

        return {key: loader(row) for key, loader in self.loaders.items()}

    def find_samples_by_column(self, column, value, collate_fn=default_collate):
        condition = self.dataframe[self.columns.index(column)] == value
        ixs = np.argwhere(condition)
        if len(ixs) == 0:
            return None

        return collate_fn([self[ix] for ix in ixs])
