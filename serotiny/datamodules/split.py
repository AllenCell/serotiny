import re
import logging
from typing import Union, Optional, Dict
from pathlib import Path

import multiprocessing as mp
import numpy as np
import pandas as pd
import pytorch_lightning as pl

from torch.utils.data.sampler import SubsetRandomSampler

from serotiny.io.dataframe import DataframeDataset, load_csv
from serotiny.datamodules.utils import ModeDataLoader

log = logging.getLogger(__name__)


def make_dataloader(
        dataset,
        mode,
        **kwargs):

    return ModeDataLoader(
        dataset=dataset,
        mode=mode,
        multiprocessing_context=mp.get_context("fork"),
        **kwargs,
    )


class SplitDatamodule(pl.LightningDataModule):
    """
    A pytorch lightning datamodule that assumes the data has already been split
    into train, valid and test manifests.

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
    pin_memory: bool = True
        Set to true when using GPU, for better performance
    drop_last: bool = False
        Whether to drop the last batch (in case the given batch size is the only
        supported)
    """

    def __init__(
        self,
        split_path: Union[Path, str],
        batch_size: int,
        num_workers: int,
        loaders: Dict,
        pin_memory: bool = True,
        drop_last: bool = False,
    ):

        super().__init__()

        split_path = Path(split_path)
        self.datasets = {}
        for csv in list(split_path.glob("*.csv")):
            key = re.findall(r'(.*)\.csv', csv.name)[0]
            dataframe = load_csv(csv)
            dataset = DataframeDataset(dataframe, loaders)
            self.datasets[key] = dataset

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last

    def train_dataloader(self):
        return make_dataloader(
            self.datasets['train'],
            "train",
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last)

    def val_dataloader(self):
        return make_dataloader(
            self.datasets['valid'],
            "eval",
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last)

    def test_dataloader(self):
        return make_dataloader(
            self.datasets['test'],
            "eval",
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last)
