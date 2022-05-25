import re
from itertools import chain
from upath import UPath as Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data import DataLoader

from serotiny.io.dataframe import DataframeDataset, read_dataframe
from serotiny.io.dataframe.loaders.abstract_loader import Loader
from .utils import FastDataLoader


class ManifestDatamodule(pl.LightningDataModule):
    """A pytorch lightning datamodule based on manifest files. It can either
    use a single manifest file, which contains a column based on which a train-
    val-test split can be made; or it can use three manifest files, one for
    each fold (train, val, test).

    Additionally, if it is only going to be used for prediction/testing, a flag
    `just_inference` can be set to True so the splits are ignored and the whole
    dataset is used.

    The `predict_datamodule` is simply constructed from the whole dataset,
    regardless of the value of `just_inference`.
    """

    def __init__(
        self,
        path: Union[Path, str],
        loaders: Union[Dict, Loader],
        split_column: Optional[Union[Path, str]] = None,
        columns: Optional[Sequence[str]] = None,
        just_inference: bool = False,
        dataloader_type: str = "fast",
        **dataloader_kwargs,
    ):
        """
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

        just_inference: bool = False
            Whether this datamodule will be used for just inference
            (testing/prediction).
            If so, the splits are ignored and the whole dataset is used.

        dataloader_kwargs:
            Additional keyword arguments are passed to the
            torch.utils.data.DataLoader class when instantiating it (aside from
            `shuffle` which is only used for the train dataloader).
            Among these args are `num_workers`, `batch_size`, `shuffle`, etc.
            See the PyTorch docs for more info on these args:
            https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
        """

        super().__init__()
        self.path = path
        self._dataloader_cls = (
            FastDataLoader if dataloader_type == "fast" else DataLoader
        )

        # if only one loader is specified, the same loaders are
        # used for all  folds
        loaders = _parse_loaders(loaders)

        path = Path(path)

        # if path is a directory, we assume it is a directory containing multiple
        # manifests - one per split. otherwise, we assume it is the path to a single
        # manifest file, which is expected to have a
        if path.is_dir():
            self.datasets = _make_multiple_manifest_splits(
                path, loaders, columns, just_inference
            )
        else:
            if split_column is None and not just_inference:
                raise MisconfigurationException(
                    "When using a single manifest file, it must have a "
                    "split column, to use for train-val-test splitting."
                )

            self.datasets = _make_single_manifest_splits(
                path, loaders, split_column, columns, just_inference
            )

        self.just_inference = just_inference
        self.dataloader_kwargs = dataloader_kwargs
        self.dataloaders = dict()

    def make_dataloader(self, split):
        kwargs = dict(**self.dataloader_kwargs)
        kwargs["shuffle"] = kwargs.get("shuffle", True) and split == "train"

        return self._dataloader_cls(dataset=self.datasets[split], **kwargs)

    def _get_dataloader(self, split):
        if split not in self.dataloaders:
            self.dataloaders[split] = self.make_dataloader(split)

        return self.dataloaders[split]

    def train_dataloader(self):
        if self.just_inference:
            raise TypeError(
                "This datamodule was configured with `just_inference=True`, "
                "so it doesn't have a train_dataloader and can't be "
                "used for training."
            )
        return self._get_dataloader("train")

    def val_dataloader(self):
        if self.just_inference:
            raise TypeError(
                "This datamodule was configured with `just_inference=True`, "
                "so it doesn't have a train_dataloader and can't be "
                "used for training."
            )

        return self._get_dataloader("val")

    def test_dataloader(self):
        split = "predict" if self.just_inference else "test"
        return self._get_dataloader(split)

    def predict_dataloader(self):
        return self._get_dataloader("predict")


def _get_canonical_split_name(split):
    for canon in ["train", "val", "test", "predict"]:
        if split.startswith(canon) or canon.startswith(split):
            return canon
    raise ValueError


def _make_single_manifest_splits(
    manifest_path, loaders, split_column, columns=None, just_inference=False
):
    dataframe = read_dataframe(manifest_path, columns)
    if not just_inference:
        assert dataframe.dtypes[split_column] == np.dtype("O")

    split_names = dataframe[split_column].unique().tolist()
    if not just_inference:
        assert set(split_names).issubset(
            {"train", "training", "valid", "val", "validation", "test", "testing"}
        )

    if split_column != "split":
        dataframe["split"] = dataframe[split_column].apply(_get_canonical_split_name)

    if not just_inference:
        datasets = {}
        for split in ["train", "val", "test"]:
            datasets[split] = DataframeDataset(
                dataframe.loc[dataframe["split"].str.startswith(split)].copy(),
                loaders[split],
            )

    datasets["predict"] = DataframeDataset(dataframe.copy(), loaders["predict"])
    return datasets


def _make_multiple_manifest_splits(
    split_path, loaders, columns=None, just_inference=False
):
    split_path = Path(split_path)
    datasets = {}
    predict_df = []

    for fpath in chain(split_path.glob("*.csv"), split_path.glob("*.parquet")):
        split = re.findall(r"(.*)\.(?:csv|parquet)", fpath.name)[0]
        split = _get_canonical_split_name(split)
        dataframe = read_dataframe(fpath, required_columns=columns)
        dataframe["split"] = split
        dataset = DataframeDataset(dataframe, loaders=loaders[split])
        if not just_inference:
            datasets[split] = dataset
        predict_df.append(dataframe.copy())

    predict_df = pd.concat(predict_df)
    datasets["predict"] = DataframeDataset(predict_df, loaders=loaders["predict"])

    return datasets


def _dict_depth(d):
    return max(_dict_depth(v) if isinstance(v, dict) else 0 for v in d.values()) + 1


def _parse_loaders(loaders):
    depth = _dict_depth(loaders)
    if depth == 1:
        loaders = {split: loaders for split in ["train", "val", "test", "predict"]}
    elif depth != 2:
        raise ValueError(f"Loaders dict should have depth 1 or 2. Got {depth}")

    for k, v in loaders.items():
        loaders[_get_canonical_split_name(k)] = v

    for k in loaders:
        if isinstance(loaders[k], str):
            assert loaders[k] in loaders
            loaders[k] = loaders[loaders[k]]

    for split in ["train", "val", "test"]:
        if split not in loaders:
            raise ValueError(f"'{split}' missing from loaders dict.")

    if "predict" not in loaders:
        loaders["predict"] = loaders["test"]

    return loaders
