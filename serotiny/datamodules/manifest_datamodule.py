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
from serotiny.utils import invoke_class

log = logging.getLogger(__name__)

def make_manifest_dataset(
    manifest: Union[Path, str],
    loaders_config: Dict,
    columns: Optional[Sequence[str]] = None,
    fms: bool = False,
    iloc: bool = False,
):
    if fms:
        manifest = Path(FileManagementSystem().get_file_by_id(manifest).path)
    else:
        manifest = Path(manifest)

    if not manifest.is_file():
        raise FileNotFoundError("Manifest file not found at given path")
    if manifest.suffix == ".csv":
        df = pd.read_csv(manifest)
    elif manifest.suffix == ".parquet":
        df = pd.read_parquet(manifest, columns=columns)
    else:
        raise TypeError("File type of provided manifest is not .csv")

    loaders = {
        key: invoke_class(value)
        for key, value in loaders_config.items()
    }

    return DataframeDataset(
        dataframe=df,
        loaders=loaders,
        iloc=iloc)


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
        loaders_config: Dict,
        split_col: Optional[str] = None,
        columns: Optional[Sequence[str]] = None,
        pin_memory: bool = True,
        drop_last: bool = False,
        metadata: Dict = {},
        subset_train: float = 1.0,
        fms: bool = False
    ):

        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = make_manifest_dataset(
            manifest,
            loaders_config['train'],
            columns,
            fms)
        self.length = len(self.dataset)

        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.metadata = metadata

        assert subset_train <= 1

        indices = list(range(self.length))
        index = {}
        if split_col is not None:
            dataframe = self.dataset.dataframe

            assert dataframe.dtypes[split_col] == np.dtype("O")
            dataframe[split_col] = dataframe[split_col].str.lower()
            split_names = dataframe[split_col].unique().tolist()
            assert set(split_names).issubset({"train", "validation", "test"})

            index_mapping = {
                'train': 'train',
                'valid': 'validation',
                'test': 'test'}
            index = {}
            for mode, value in index_mapping.items():
                index[mode] = dataframe.loc[
                    dataframe[split_col] == value
                ].index.tolist()
        else:
            index['train'] = indices
            index['valid'] = [0] * self.batch_size
            index['test'] = [0] * self.batch_size

        if subset_train < 1:
            new_size = int(subset_train * len(index['train']))

            log.info(
                f"Subsetting the training data by {100*subset_train:.2f}%, "
                f"from {len(train_idx)} to {new_size}")

            index['train'] = np.random.choice(
                index['train'],
                replace=False,
                size=new_size)

        self.samplers = {}
        self.datasets = {}

        for mode in index:
            self.samplers[mode] = SubsetRandomSampler(index[mode])
            self.datasets[mode] = make_manifest_dataset(
                manifest,
                loaders_config[mode],
                columns,
                fms)
        
    def make_dataloader(self, mode):
        return DataLoader(
            dataset=self.datasets[mode],
            sampler=self.samplers[mode],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last)

    def train_dataloader(self):
        return self.make_dataloader("train")

    def val_dataloader(self):
        return self.make_dataloader("valid")

    def test_dataloader(self):
        return self.make_dataloader("test")
