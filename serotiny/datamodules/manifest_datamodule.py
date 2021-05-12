from typing import Union, Optional, Dict
from pathlib import Path

import json
import numpy as np
import pandas as pd
import multiprocessing as mp
import pytorch_lightning as pl

from torch.utils.data.sampler import SubsetRandomSampler

from serotiny.io.dataframe import DataframeDataset
from serotiny.utils import get_classes_from_config
from serotiny.datamodules.utils import TrainDataLoader, EvalDataLoader

def make_manifest_dataset(
    manifest: Union[Path, str],
    loader_dict: Dict,
    split_col: Optional[str] = None,
):
    manifest = Path(manifest)
    if not manifest.is_file():
        raise FileNotFoundError("Manifest file not found at given path")
    if not manifest.suffix == ".csv":
        raise TypeError("File type of provided manifest is not .csv")

    df = pd.read_csv(manifest)

    loaders = {
        key: get_classes_from_config(value)[0]
        for key, value in loader_dict.items()
    }

    return DataframeDataset(dataframe=df, loaders=loaders,
                            iloc=True, split_col=split_col)

def make_dataloader(dataset, batch_size, num_workers, sampler, pin_memory,
                    stage, drop_last=False):
    if stage == "train":
        dataloader_class = TrainDataLoader
    else:
        dataloader_class = EvalDataLoader

    return dataloader_class(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=drop_last,
        num_workers=num_workers,
        multiprocessing_context=mp.get_context("fork"),
        sampler=sampler,
    )


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
    pin_memory: bool = True
        Set to true when using GPU, for better performance
    drop_last: bool = False
        Whether to drop the last batch (in case the given batch size is the only
        supported)

    """

    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        manifest: Union[Path, str],
        loader_dict: Dict,
        split_col: Optional[str] = None,
        pin_memory: bool = True,
        drop_last: bool = True,
        metadata: dict,
    ):

        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = make_manifest_dataset(manifest, loader_dict, split_col)
        self.length = len(self.dataset)

        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.metadata = metadata

        indices = list(range(self.length))
        if split_col is not None:
            train_idx = self.dataset.train_split
            val_idx = self.dataset.val_split
            test_idx = self.dataset.test_split
        else:
            train_idx = indices
            val_idx = [0] * self.batch_size
            test_idx = [0] * self.batch_size

        self.train_sampler = SubsetRandomSampler(train_idx)
        self.val_sampler = SubsetRandomSampler(val_idx)
        self.test_sampler = SubsetRandomSampler(test_idx)

    def train_dataloader(self):
        return make_dataloader(self.dataset, self.batch_size, self.num_workers,
                               self.train_sampler, self.pin_memory, "train",
                               self.drop_last)

    def val_dataloader(self):
        return make_dataloader(self.dataset, self.batch_size, self.num_workers,
                               self.val_sampler, self.pin_memory, "eval",
                               self.drop_last)

    def test_dataloader(self):
        return make_dataloader(self.dataset, self.batch_size, self.num_workers,
                               self.test_sampler, self.pin_memory, "eval",
                               self.drop_last)
