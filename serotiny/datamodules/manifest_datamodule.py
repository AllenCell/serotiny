import logging
from typing import Union, Optional, Dict, Sequence
from pathlib import Path

import multiprocessing as mp
import numpy as np
import pandas as pd
import pytorch_lightning as pl

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

from serotiny.io.dataframe import DataframeDataset
from serotiny.utils import load_multiple

# from aicsfiles import FileManagementSystem

log = logging.getLogger(__name__)

def _read_dataframe(
    manifest: Union[Path, str],
    columns: Optional[Sequence[str]] = None,
):
    manifest = Path(manifest)

    if not manifest.is_file():
        raise FileNotFoundError("Manifest file not found at given path")
    if manifest.suffix == ".csv":
        df = pd.read_csv(manifest)
        if columns is not None:
            df = df[columns]
    elif manifest.suffix == ".parquet":
        df = pd.read_parquet(manifest, columns=columns)
    else:
        raise TypeError("File type of provided manifest is not .csv")

    return df


class ManifestDatamodule(pl.LightningDataModule):
    """
    A pytorch lightning datamodule that handles the logic for iterating over a
    folder of files

    Parameters
    -----------
    batch_size: int
        batch size for dataloader
    num_workers: int
        Number of worker processes for dataloader
    manifest: Optional[Union[Path, str]] = None
        (optional) Path to a manifest file to be merged with the folder, or to
        be the core of this dataset, if no path is provided
    loader_dict: Dict
        Dictionary of loader specifications for each given key. When the value
        is callable, that is the assumed loader. When the value is a tuple, it
        is assumed to be of the form (loader class name, loader class args) and
        will be used to instantiate the loader
    split_col: Optional[str] = None
        Name of a column in the dataset which can be used to create train, val, test
        splits.
    columns: Optional[Sequence[str]] = None
        List of columns to load from the dataset, in case it's a parquet file.
        If None, load everything.
    pin_memory: bool = True
        Set to true when using GPU, for better performance
    drop_last: bool = False
        Whether to drop the last batch (in case the given batch size is the only
        supported)
    subset_train: float = 1.0
    """

    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        manifest: Union[Path, str],
        loaders: Dict,
        split_col: Optional[str] = None,
        columns: Optional[Sequence[str]] = None,
        pin_memory: bool = True,
        drop_last: bool = False,
        metadata: Dict = {},
        subset_train: float = 1.0,
    ):

        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.metadata = metadata

        assert subset_train <= 1

        self.dataframe = _read_dataframe(manifest, columns)
        self.length = len(self.dataframe)
        indices = list(range(self.length))

        loaders["train"] = load_multiple(loaders["train"])
        for split in ["validation", "test"]:
            if split in loaders:
                loaders[split] = load_multiple(loaders[split])
            else:
                loaders[split] = loaders["train"]

        indices = {}
        if split_col is not None:
            assert self.dataframe.dtypes[split_col] == np.dtype("O")
            self.dataframe[split_col] = self.dataframe[split_col].str.lower()
            split_names = self.dataframe[split_col].unique().tolist()
            assert set(split_names).issubset({"train", "validation", "test"})

            for split in ["train", "validation", "test"]:
                indices[split] = self.dataframe.loc[
                    self.dataframe[split_col] == split
                ].index.tolist()
        else:
            indices['train'] = self.dataframe.index.tolist()
            indices['validation'] = [0] * self.batch_size
            indices['test'] = [0] * self.batch_size

        if subset_train < 1:
            new_size = int(subset_train * len(indices['train']))

            log.info(
                f"Subsetting the training data by {100*subset_train:.2f}%, "
                f"from {len(indices['train'])} to {new_size}")

            indices['train'] = np.random.choice(
                indices['train'],
                replace=False,
                size=new_size)

        self.samplers = {}
        self.datasets = {}

        for split in indices:
            self.samplers[split] = SubsetRandomSampler(indices[split])
            self.datasets[split] = DataframeDataset(self.dataframe, loaders[split])

    def make_dataloader(self, split):
        return DataLoader(
            dataset=self.datasets[split],
            sampler=self.samplers[split],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last)

    def train_dataloader(self):
        return self.make_dataloader("train")

    def val_dataloader(self):
        return self.make_dataloader("validation")

    def test_dataloader(self):
        return self.make_dataloader("test")
