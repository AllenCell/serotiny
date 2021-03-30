from typing import Union, Optional, List, Dict, Callable, Tuple
from pathlib import Path
from collections import defaultdict

import copy

import numpy as np
import pandas as pd
import multiprocessing as mp
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from ..io.dataframe_dataset import DataframeDataset
from ..io.loaders import infer_extension_loader, LoadColumns

def _unnest_list(l):
    res = []
    for obj in l:
        if isinstance(obj, tuple):
            res.extend(obj)
        else:
            res.append(obj)
    return res

def _validate_manifest(manifest, path_col):
    if not manifest.is_file():
        raise FileNotFoundError("Manifest file not found at given path")
    if not manifest.suffix == ".csv":
        raise TypeError("File type of provided manifest is not .csv")
    if path_col is None:
        raise ValueError("If using a manifest file, `path_col` cannot be None")

def infer_extension(
    list_of_files: List[Path],
    return_files: bool=True
):
    """
    Util function to infer what is the file extension to be used for a folder
    dataset, given a list of file paths.

    Parameters
    ----------
    list_of_files: List[Path]
        list of file paths in the folder
    return_files: bool
        flag to determine whether to return the list of files with the inferred
        extension. if false, return only the extension as a string
    """
    if return_files:
        extension_files = defaultdict(list)
        max_func = lambda k: len(extension_files[k])
    else:
        extension_files = defaultdict(int)
        max_func = lambda k: extension_files[k]

    for f in list_of_files:
        if f.is_file():
            if return_files:
                extension_files[f.suffix].append(f)
            else:
                extension_files[f.suffix] += 1

    inferred_extension = max(extension_files.keys(), key=max_func)

    if return_files:
        return inferred_extension, extension_files[inferred_extension]
    return inferred_extension

def _resolve_loaders(loader_dict):
    for loader_key, loader in loader_dict.items():
        if isinstance(loader, tuple):
            loader_name, loader_args = loader
            loader_class = getattr(loaders, loader_name)
            loader_dict[loader_key] = loader_class(**loader_args)
    return loader_dict

def make_folder_dataset(
    path: Optional[Union[Path, str]] = None,
    extension: Optional[str] = None,
    manifest: Optional[Union[Path, str]] = None,
    path_col: Optional[str] = None,
    loader_dict: Dict[str, Union[Callable, Dict[str, Tuple[str, Dict]]]] = {},
    split_col: Optional[str] = None,
    x_label: str = "images",
    return_paths: bool = False,
):
    loader_dict = copy.copy(loader_dict)
    loader_dict = _resolve_loaders(loader_dict)
    if (path is None) and (manifest is None):
        raise ValueError("Either `path` or `manifest` must be specified")

    df = None
    if path is not None:
        path = Path(path)
        assert path.is_dir()

        if extension is None:
            list_of_files = path.glob("*")
            extension, list_of_files = infer_extension(list_of_files, return_files=True)
        else:
            if extension[0] != ".":
                extension = "." + extension
                list_of_files = path.glob(f"*{extension}")

        df = pd.Series({f.name: f for f in list_of_files},
                        name="true_paths").to_frame()
        df["basename"] = df.index.values

        loader_dict[x_label] = infer_extension_loader(extension)
        if return_paths:
            loader_dict["paths"] = LoadColumns(["basename"])

    manifest_df = None
    if manifest is not None:
        manifest = Path(manifest)
        _validate_manifest(manifest, path_col)

        manifest_df = pd.read_csv(manifest)
        manifest_df["basename"] = manifest_df[path_col].apply(lambda f: Path(f).name)
        if return_paths:
            loader_dict["paths"] = LoadColumns(["basename"])

    if (df is not None) and (manifest_df is not None):
        df = df.merge(right=manifest_df, how="left", left_index=True, right_on="basename")
    elif manifest_df is not None:
        df = manifest_df

    return DataframeDataset(dataframe=df, loaders=loader_dict, iloc=True, split_col=split_col)

def make_dataloader(dataset, batch_size, num_workers, sampler):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=True,
        num_workers=num_workers,
        multiprocessing_context=mp.get_context("fork"),
        sampler=sampler,
    )

def train_val_test_split(indices, train_frac, test_frac=0.5, **kwargs):
    train_split, val_test = train_test_split(indices, train_size=train_frac, **kwargs)
    val_split, test_split = train_test_split(val_test, test_size=test_frac)

    return train_split, val_split, test_split

class FolderDatamodule(pl.LightningDataModule):
    """
    A pytorch lightning datamodule that handles the logic for iterating over a
    folder of files

    Parameters
    -----------
    batch_size: int
        batch size for dataloader
    num_workers: int
        Number of worker processes for dataloader
    path: Optional[Union[Path, str]]
        Path to the folder around which this Datamodule will wrap. If None, will
        assume a manifest is passed, from which the dataset is obtained
    extension: Optional[str] = None
        Extension of the files to be considered as the dataset entries. If None,
        the most common extension in the provided folder will be assumed
    manifest: Optional[Union[Path, str]] = None
        (optional) Path to a manifest file to be merged with the folder, or to
        be the core of this dataset, if no path is provided
    path_col: Optional[str] = None
        Name of the manifest columns which should contain paths. If a folder path
        was passed as `path`, the basename of these paths should match that of
        the files in the folder.
    loader_dict: Dict[str, Union[Callable, Dict[str, Tuple[str, Dict]]]] = {}
        Dictionary of loader specifications for each given key. When the value
        is callable, that is the assumed loader. When the value is a tuple, it
        is assumed to be of the form (loader class name, loader class args) and
        will be used to instantiate the loader
    split_col: Optional[str] = None
        Name of a column in the dataset which can be used to create train, val, test
        splits.
    train_frac: float = 1.0,
        Fraction of samples in the dataset to be used for training. If `split_col`
        is given, `train_frac` is ignored.
    test_set: bool = False
        Flag to determine whether to create a test set. If `split_col` is given,
        `test_set` is ignored.
    seed: int = 42,
        Seed used to make splits reproducible.
    return_paths: bool = False,
        Flag to determine whether to return the file paths in the generated batches
    """

    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        path: Optional[Union[Path, str]] = None,
        extension: Optional[str] = None,
        manifest: Optional[Union[Path, str]] = None,
        path_col: Optional[str] = None,
        loader_dict: Dict[str, Union[Callable, Dict[str, Tuple[str, Dict]]]] = {},
        split_col: Optional[str] = None,
        train_frac: float = 1.0,
        test_set: bool = False,
        seed: int = 42,
        x_label: str = "images",
        return_paths: bool = False,
    ):

        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = make_folder_dataset(path, extension, manifest, path_col,
                                           loader_dict, split_col, x_label, return_paths)
        self.length = len(self.dataset)

        indices = list(range(self.length))
        np.random.seed(seed)
        if split_col is not None:
            train_idx = self.dataset.train_split
            val_idx = self.dataset.val_split
            test_idx = self.dataset.test_split
        elif train_frac < 1:
            if test_set:
                train_idx, val_idx, test_idx = train_val_test_split(indices, train_frac)
            else:
                train_idx, val_idx = train_test_split(indices, train_frac)
                test_idx = []
        else:
            train_idx = indices
            val_idx = []
            test_idx = []

        self.train_sampler = SubsetRandomSampler(train_idx)
        self.val_sampler = SubsetRandomSampler(val_idx)
        self.test_sampler = SubsetRandomSampler(test_idx)

    def train_dataloader(self):
        return make_dataloader(self.dataset, self.batch_size, self.num_workers,
                               self.train_sampler)

    def val_dataloader(self):
        return make_dataloader(self.dataset, self.batch_size, self.num_workers,
                               self.val_sampler)

    def test_dataloader(self):
        return make_dataloader(self.dataset, self.batch_size, self.num_workers,
                               self.test_sampler)
