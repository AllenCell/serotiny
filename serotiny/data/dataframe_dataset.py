from collections.abc import Iterable
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate as collate


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
    transform: callable
        A transform to be applied to the items of this dataset when they are
        loaded
    """

    def __init__(self, dataframe, loaders=None, transform=None, iloc=True):
        self.dataframe = dataframe
        self.loaders = loaders
        self.transform = transform
        self.iloc = iloc

    def __len__(self):
        return len(self.dataframe)

    def _get_single_item(self, idx):
        if self.iloc:
            row = self.dataframe.iloc[idx, :]
        else:
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
