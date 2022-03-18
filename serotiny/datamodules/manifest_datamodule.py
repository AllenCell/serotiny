import re
from itertools import chain
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Union

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data import DataLoader

from serotiny.io.dataframe import DataframeDataset, read_dataframe
from serotiny.io.dataframe.loaders.abstract_loader import Loader


def _make_single_manifest_splits(manifest_path, loaders, split_column, columns=None):
    dataframe = read_dataframe(manifest_path, columns)
    assert dataframe.dtypes[split_column] == np.dtype("O")

    split_names = dataframe[split_column].unique().tolist()
    assert set(split_names).issubset({"train", "valid", "test"})

    return {
        split: DataframeDataset(
            dataframe.loc[dataframe[split_column] == split].copy(), loaders[split]
        )
        for split in ["train", "valid", "test"]
    }


def _make_multiple_manifest_splits(split_path, loaders, columns=None):
    split_path = Path(split_path)
    datasets = {}

    for fpath in chain(split_path.glob("*.csv"), split_path.glob("*.parquet")):
        mode = re.findall(r"(.*)\.(?:csv|parquet)", fpath.name)[0]
        dataframe = read_dataframe(fpath, required_columns=columns)
        dataset = DataframeDataset(dataframe, loaders=loaders[mode])
        datasets[mode] = dataset

    return datasets


def _dict_depth(d):
    return max(_dict_depth(v) if isinstance(v, dict) else 0 for v in d.values()) + 1


def _parse_loaders(loaders):
    depth = _dict_depth(loaders)
    if depth == 1:
        loaders = {split: loaders for split in ["train", "valid", "test"]}
    elif depth != 2:
        raise ValueError(f"Loaders dict should have depth 1 or 2. Got {depth}")

    return loaders


class ManifestDatamodule(pl.LightningDataModule):
    """A pytorch lightning datamodule based on manifest files. It can either use a
    single manifest file, which contains a column based on which a train- val-test split
    can be made; or it can use three manifest files, one for each fold (train, val,
    test)

    Parameters
    ----------
    path: Union[Path, str]
        Path to a manifest file

    batch_size: int
        batch size for dataloader

    num_workers: int
        Number of worker processes for dataloader

    loaders: Union[Dict, Loader]
        Loader specifications for each given split.

    split_column: Optional[str] = None
        Name of a column in the dataset which can be used to create train, val, test
        splits.

    columns: Optional[Sequence[str]] = None
        List of columns to load from the dataset, in case it's a parquet file.
        If None, load everything.

    collate: Optional[Callable] = None
        Alternative collate function for dataloader

    pin_memory: bool = True
        Set to true when using GPU, for better performance

    drop_last: bool = False
        Whether to drop the last batch (in case the given batch size is the only
        supported)
    """

    def __init__(
        self,
        path: Union[Path, str],
        batch_size: int,
        num_workers: int,
        loaders: Union[Dict, Loader],
        split_column: Optional[Union[Path, str]] = None,
        columns: Optional[Sequence[str]] = None,
        collate: Optional[Callable] = None,
        pin_memory: bool = True,
        shuffle: bool = True,
        drop_last: bool = False,
        multiprocessing_context: Optional[str] = None,
    ):

        super().__init__()
        self.path = path
        self.multiprocessing_context = multiprocessing_context

        # if only one loader is specified, the same loaders are
        # used for all  folds
        loaders = _parse_loaders(loaders)

        path = Path(path)

        # if path is a directory, we assume it is a directory containing multiple
        # manifests - one per split. otherwise, we assume it is the path to a single
        # manifest file, which is expected to have a
        if path.is_dir():
            self.datasets = _make_multiple_manifest_splits(path, loaders, columns)
        else:
            if split_column is None:
                raise MisconfigurationException(
                    "When using a single manifest file, it must have a "
                    "split column, to use for train-val-test splitting."
                )

            self.datasets = _make_single_manifest_splits(
                path, loaders, split_column, columns
            )

        # DataLoader arguments
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.collate = collate

        self._predict_split = "test"

    def make_dataloader(self, split, shuffle=False):
        shuffle = self.shuffle or shuffle
        return DataLoader(
            dataset=self.datasets[split],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            collate_fn=self.collate,
            multiprocessing_context=self.multiprocessing_context,
            shuffle=(shuffle if split == "train" else False),
        )

    def train_dataloader(self):
        return self.make_dataloader("train")

    def val_dataloader(self):
        return self.make_dataloader("valid")

    def test_dataloader(self):
        return self.make_dataloader("test")

    def predict_dataloader(self):
        return self.make_dataloader(self._predict_split, shuffle=False)
