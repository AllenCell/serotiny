from typing import Sequence
from pathlib import Path

import pandas as pd
import multiprocessing as mp
import pytorch_lightning as pl

from torch.utils.data import DataLoader

from serotiny.utils.dynamic_imports import load_multiple
from serotiny.io.buffered_patch_dataset import BufferedPatchDataset
from serotiny.io.dataframe import DataframeDataset

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
        super().__init__()

        self.manifest_path = Path(manifest_path)

        self.datasets = {}
        self.loaders = {}

        for mode, loaders_config in loaders.items():
            self.loaders[mode] = load_multiple(loaders_config)

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

    def load_patch_manifest(self, mode):
        manifest = make_manifest_dataset(
            self.manifest_path / f'{mode}.csv',
            self.loaders[mode])
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
