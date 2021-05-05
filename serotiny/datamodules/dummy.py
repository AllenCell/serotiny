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

    def __init__(self, x_label, y_label, length, dims):
        self.length = length
        self.dims = tuple(dims)
        self.x_label = x_label
        self.y_label = y_label

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            self.x_label: torch.randn(self.dims),
            self.y_label: torch.randint(high=10, size=(1,)),
        }


def make_dataloader(x_label, y_label, length, dims, batch_size, num_workers):
    """
    Instantiate dummy dataset and return dataloader
    """
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
        dims: list,
        length: int,
        channels: list = [],
    ):

        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.x_label = x_label
        self.y_label = y_label
        self.length = length

        self.num_channels = len(channels)
        self.dims = dims

        if self.num_channels > 0:
            dl_dims = tuple([self.num_channels] + list(dims)),
        else:
            dl_dims = dims

        self.dataloader = make_dataloader(
            x_label,
            y_label,
            length,
            dl_dims,
            batch_size,
            num_workers,
        )

    def train_dataloader(self):
        return self.dataloader

    def val_dataloader(self):
        return self.dataloader

    def test_dataloader(self):
        return self.dataloader
