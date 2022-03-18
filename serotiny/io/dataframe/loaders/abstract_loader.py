class Loader:
    """Abstract class for manifest loaders. Subclasses of this class shall implement
    specific mechanisms to load data from pandas dataframes.

    Instances of this class (and its subclasses) are callable objects, used by the
    `DataframeDataset` class, which always provides them with a dataframe row from which
    data is extracted
    """

    def __init__(self):
        pass

    def __call__(self, row):
        raise NotImplementedError
