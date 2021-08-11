from torch.utils.data import Dataset


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
    """

    def __init__(self, dataframe, loaders=None, iloc=True):
        self.dataframe = dataframe
        if iloc:
            self.dataframe = self.dataframe.reset_index()

        self.loaders = loaders
        self.iloc = iloc

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if self.iloc:
            row = self.dataframe.iloc[idx]
        else:
            row = self.dataframe.loc[idx]

        return {key: loader(row) for key, loader in self.loaders.items()}
