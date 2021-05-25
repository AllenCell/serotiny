from typing import Union, Optional, Dict, Sequence
from pathlib import Path

import json
import numpy as np
import pandas as pd
import multiprocessing as mp
import pytorch_lightning as pl

from torch.utils.data import DataLoader

from serotiny.utils import get_classes_from_config, get_class_from_path, path_invocations
from serotiny.io.buffered_patch_dataset import BufferedPatchDataset


def make_manifest_dataset(
        manifest: str,
        loaders: dict):
    dataframe = pd.read_csv(manifest)

    return DataframeDataset(
        dataframe=dataframe,
        loaders=loaders,
        iloc=True)


class PatchDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        manifest_path: str,
        loaders: dict,
        batch_size: int,
        num_workers: int,
        pin_memory: bool = True,
        drop_last: bool = True,
        patch_columns: Sequence[str] = None,
        patch_shape: Sequence[int] = (32, 64, 64),
        buffer_size: int = 1,
        buffer_switch_interval: int = -1,
        shuffle_images: bool = True,
    ):
        self.manifest_path = Path(manifest_path)
        self.loaders = path_invocations(loaders)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        self.patch_columns = patch_columns
        self.patch_shape = patch_shape
        self.buffer_size = buffer_size
        self.buffer_switch_interval = buffer_switch_interval
        self.shuffle_images = shuffle_images

        self.train = self.load_patch_manifest('train')
        self.valid = self.load_patch_manifest('valid')
        self.test = self.load_patch_manifest('test')

    def make_patch_dataset(self, dataset):
        return BufferedPatchDataset(
            dataset=dataset,
            patch_columns=self.patch_columns,
            patch_shape=self.patch_shape,
            buffer_size=self.buffer_size,
            buffer_switch_interval=self.buffer_switch_interval,
            shuffle_images=self.shuffle_images)

    def load_patch_manifest(self, manifest_key):
        manifest = make_manifest_dataset(
            self.manifest_path / f'{manifest_key}.csv',
            self.loaders)
        return self.make_patch_dataset(manifest)

    def make_dataloader(self, dataset):
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            num_workers=self.num_workers,
            multiprocessing_context=mp.get_context("fork"))

    def train_dataloader(self):
        return self.make_dataloader(self.train)

    def val_dataloader(self):
        return self.make_dataloader(self.valid)

    def test_dataloader(self):
        return self.make_dataloader(self.test)
