from typing import Union, Optional, Dict
from pathlib import Path

import json
import numpy as np
import pandas as pd
import multiprocessing as mp
import pytorch_lightning as pl

from torch.utils.data import DataLoader

from serotiny.utils import get_classes_from_config
from serotiny.io.buffered_patch_dataset import BufferedPatchDataset


make_patch_dataset(manifest_path):




class PatchDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        manifest_path: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool = True,
        drop_last: bool = True,
    ):
        self.manifest_path = Path(manifest_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        self.train = make_patch_dataset(self.manifest_path / 'train.csv')
        self.valid = make_patch_dataset(self.manifest_path / 'valid.csv')
        self.test = make_patch_dataset(self.manifest_path / 'test.csv')

    def make_dataloader(self, dataset):
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            num_workers=self.num_workers,
            multiprocessing_context=mp.get_context("fork"),
        )

    def train_dataloader(self):
        return self.make_dataloader(self.train)

    def val_dataloader(self):
        return self.make_dataloader(self.valid)

    def test_dataloader(self):
        return self.make_dataloader(self.test)
