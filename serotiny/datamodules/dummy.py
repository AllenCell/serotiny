import multiprocessing as mp
import pytorch_lightning as pl

import torch
from torch.utils.data import Dataset, DataLoader

class DummyDataset(Dataset):
    def __init__(self, x_label, y_label, length, dims, *args, **kwargs):
        self.length = length
        self.dims = dims
        self.x_label = x_label
        self.y_label = y_label

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            self.x_label: torch.randn(self.dims),
            self.y_label: torch.randint(high=10, size=(1,))
        }

def make_dataloader(x_label, y_label, length, dims, batch_size, num_workers):
    dataset = DummyDataset(x_label, y_label, length, dims)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=True,
        num_workers=num_workers,
        multiprocessing_context=mp.get_context("fork"),
    )

class DummyDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        x_label: str,
        y_label: str,
        dims: list,
        length: int,
        channels: list = [],
        **kwargs
    ):

        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.x_label = x_label
        self.y_label = y_label
        self.length = length

        n_channels = len(channels)
        self.dims = dims

        self.dataloader = make_dataloader(
            x_label,
            y_label,
            length,
            tuple([n_channels]+dims),
            batch_size,
            num_workers
        )

    def train_dataloader(self):
        return self.dataloader

    def val_dataloader(self):
        return self.dataloader

    def test_dataloader(self):
        return self.dataloader
