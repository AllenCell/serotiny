import re
from itertools import chain
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
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

    datasets = {
        split: DataframeDataset(
            dataframe.loc[dataframe[split_column] == split].copy(), loaders[split]
        )
        for split in ["train", "valid", "test"]
    }

    datasets["predict"] = DataframeDataset(dataframe.copy(), loaders["predict"])
    return datasets


def _make_multiple_manifest_splits(split_path, loaders, columns=None):
    split_path = Path(split_path)
    datasets = {}
    predict_df = []

    for fpath in chain(split_path.glob("*.csv"), split_path.glob("*.parquet")):
        mode = re.findall(r"(.*)\.(?:csv|parquet)", fpath.name)[0]
        dataframe = read_dataframe(fpath, required_columns=columns)
        dataset = DataframeDataset(dataframe, loaders=loaders[mode])
        datasets[mode] = dataset
        predict_df.append(dataframe.copy())

    predict_df = pd.concat(predict_df)
    datasets["predict"] = DataframeDataset(predict_df, loaders=loaders["predict"])

    return datasets


def _dict_depth(d):
    return max(_dict_depth(v) if isinstance(v, dict) else 0 for v in d.values()) + 1


def _parse_loaders(loaders):
    depth = _dict_depth(loaders)
    if depth == 1:
        loaders = {split: loaders for split in ["train", "valid", "test", "predict"]}
    elif depth != 2:
        raise ValueError(f"Loaders dict should have depth 1 or 2. Got {depth}")

    for k in loaders:
        if isinstance(loaders[k], str):
            assert loaders[k] in loaders
            loaders[k] = loaders[loaders[k]]

    for split in ["train", "valid", "test"]:
        if split not in loaders:
            raise ValueError(f"'{split}' missing from loaders dict.")

    if "predict" not in loaders:
        loaders["predict"] = loaders["test"]

    return loaders


class ManifestDatamodule(pl.LightningDataModule):
    """A pytorch lightning datamodule based on manifest files. It can either
    use a single manifest file, which contains a column based on which a train-
    val- test split can be made; or it can use three manifest files, one for
    each fold (train, val, test)

    Parameters
    ----------
    path: Union[Path, str]
        Path to a manifest file

    loaders: Union[Dict, Loader]
        Loader specifications for each given split.

    split_column: Optional[str] = None
        Name of a column in the dataset which can be used to create train, val, test
        splits.

    columns: Optional[Sequence[str]] = None
        List of columns to load from the dataset, in case it's a parquet file.
        If None, load everything.

    dataloader_kwargs:
        Additional keyword arguments are passed to the
        torch.utils.data.DataLoader class when instantiating it (aside from
        `shuffle` which is only used for the train dataloader).
        Among these args are `num_workers`, `batch_size`, `shuffle`, etc.
        See the PyTorch docs for more info on these args:
        https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    """

    def __init__(
        self,
        path: Union[Path, str],
        loaders: Union[Dict, Loader],
        split_column: Optional[Union[Path, str]] = None,
        columns: Optional[Sequence[str]] = None,
        **dataloader_kwargs,
    ):

        super().__init__()
        self.path = path

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

        self.dataloader_kwargs = dataloader_kwargs

    def make_dataloader(self, split):
        kwargs = dict(**self.dataloader_kwargs)
        kwargs["shuffle"] = kwargs.get("shuffle", False) and split == "train"

        return DataLoader(dataset=self.datasets[split], **kwargs)

    def train_dataloader(self):
        return self.make_dataloader("train")

    def val_dataloader(self):
        return self.make_dataloader("valid")

    def test_dataloader(self):
        return self.make_dataloader("test")

    def predict_dataloader(self):
        return self.make_dataloader("predict")
