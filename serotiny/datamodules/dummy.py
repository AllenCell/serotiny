from typing import Optional, List
import multiprocessing as mp
import pytorch_lightning as pl

import torch
from torch.utils.data import Dataset, DataLoader


class DummyDataset(Dataset):
    """
    Instantiate a dummy pytorch dataset

    Parameters:
    x_label: str
        Key to retrieve image

    y_label: str
        Key to retrieve image label

    dims: str
        Dimensions of image

    length:
        Length of the dummy dataset

    """

    def __init__(self, x_label, y_label, length, x_dims, y_dims=None):
        self.length = length
        self.x_dims = tuple(x_dims)
        self.y_dims = (tuple(y_dims) if y_dims is not None else None)
        self.x_label = x_label
        self.y_label = y_label

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.y_dims is not None:
            y = torch.randn(self.y_dims)
        else:
            y = torch.randint(high=10, size=(1,))

        return {
            self.x_label: torch.randn(self.x_dims),
            self.y_label: y,
        }


def make_dataloader(x_label, y_label, length, x_dims, batch_size, num_workers,
                    y_dims=None):
    """
    Instantiate dummy dataset and return dataloader
    """
    dataset = DummyDataset(x_label, y_label, length, x_dims,
                           y_dims=y_dims)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=True,
        num_workers=num_workers,
        multiprocessing_context=mp.get_context("fork"),
    )


class DummyDatamodule(pl.LightningDataModule):
    """
    A pytorch lightning datamodule that handles the logic for
    loading a dummy dataset

    Parameters
    -----------
    batch_size: int
        batch size for dataloader

    num_workers: int
        Number of worker processes for dataloader

    x_label: str
        x_label key to retrive image

    y_label: str
        y_label key to retrieve image label

    dims: list
        Dimensions for dummy images

    length: int
        Length of dummy dataset

    channels: list = [],
        Number of channels for dummy images
    """

    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        x_label: str,
        y_label: str,
        x_dims: list,
        length: int,
        channels: list = [],
        y_dims: Optional[List] = None,
    ):

        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.x_label = x_label
        self.y_label = y_label
        self.length = length

        self.num_channels = len(channels)
        self.x_dims = x_dims

        if self.num_channels > 0:
            dl_dims = tuple([self.num_channels] + list(x_dims)),
        else:
            dl_dims = x_dims

        self.dataloader = make_dataloader(
            x_label,
            y_label,
            length,
            dl_dims,
            batch_size,
            num_workers,
            y_dims=y_dims
        )

    def train_dataloader(self):
        return self.dataloader

    def val_dataloader(self):
        return self.dataloader

    def test_dataloader(self):
        return self.dataloader
