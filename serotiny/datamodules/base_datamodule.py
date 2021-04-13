#!/usr/bin/env python3
from os import listdir
from pathlib import Path
from typing import Sequence, Callable, Union

import pytorch_lightning as pl
from torchvision import transforms
from ..io import load_data_loader
from ..io.data import subset_channels, load_csv


class BaseDataModule(pl.LightningDataModule):
    """
    A pytorch lightning datamodule that handles the logic for
    loading a dataframe (for val and test splits) by creating
    a pytorch dataloader. NOT MEANT to be used on its own (
    logic for defining loaders is not in this class), this
    class is inherited in datamodules like act2d and actk3d

    Parameters
    -----------
    transform_list: Sequence[Callable]
        List of transforms to apply to val and test images

    train_transform_list: Sequence[Callable]
        List of transforms to apply to train images

    x_label: str
        Column name used to load an image (x)

    y_label: str
        Column name used to load the image label (y)

    batch_size: int
        Batch size for the dataloader

    num_workers: int
        Number of worker processes to create in dataloader

    id_fields: Sequence[Union[str, int]]
        Id column name for loader

    channels: Sequence[Union[str, int]]
        List of channels in the images

    select_channels: Sequence[Union[str, int]]
        List of channels to subset the original channel list

    data_dir: str
        Path to data folder containing csv's for train, val,
        and test splits
    """

    def __init__(
        self,
        transform_list: Sequence[Callable],
        train_transform_list: Sequence[Callable],
        x_label: str,
        y_label: str,
        batch_size: int,
        num_workers: int,
        id_fields: Sequence[Union[str, int]],
        channels: Sequence[Union[str, int]],
        select_channels: Sequence[Union[str, int]],
        data_dir: str,
        **kwargs,
    ):
        super().__init__()

        self.channels = channels
        self.select_channels = select_channels
        self.num_channels = len(self.channels)

        self.channel_indexes, self.num_channels = subset_channels(
            channel_subset=self.select_channels,
            channels=self.channels,
        )

        self.data_dir = data_dir

        self.id_fields = id_fields
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.datasets = {}

        self.transform = transforms.Compose(transform_list)
        self.train_transform = transforms.Compose(train_transform_list)

        self.x_label = x_label
        self.y_label = y_label

    def prepare_data(self):
        """
        Whether to download data from somewhere
        Here, we assume that data is located in data_dir
        """
        pass

    def load_image(self, dataset):
        raise NotImplementedError(
            "`load_image` hasn't been defined for this data module"
        )

    def get_dims(self, shape):
        raise NotImplementedError("`get_dims` hasn't been defined for this data module")

    def setup(self, stage=None):
        """
        Setup train, val and test dataframes. Get image dimensions
        """

        filenames = listdir(self.data_dir)
        dataset_splits = [split for split in filenames if split.endswith(".csv")]
        dataset_paths = [Path(self.data_dir + split) for split in dataset_splits]

        for path in dataset_paths:
            if not path.exists():
                raise Exception(f"not all datasets are present, missing {path}")

        for split_csv, path in zip(dataset_splits, dataset_paths):
            split = split_csv.split(".")[0]

            dataset = load_csv(path, [])

            self.datasets[split] = dataset

        # Load a test image to get image dimensions after transform
        test_image = self.load_image(dataset)

        # Get test image dimensions
        dimensions = self.get_dims(test_image)

        self.dims = dimensions

    def val_dataloader(self):
        """
        Instantiate val dataloader
        """
        val_dataset = self.datasets["valid"]
        val_dataloader = load_data_loader(
            val_dataset,
            self.loaders,
            transform=self.transform,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        return val_dataloader

    def test_dataloader(self):
        """
        Instantiate test dataloader
        """
        test_dataset = self.datasets["test"]
        test_dataloader = load_data_loader(
            test_dataset,
            self.loaders,
            transform=self.transform,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        return test_dataloader
