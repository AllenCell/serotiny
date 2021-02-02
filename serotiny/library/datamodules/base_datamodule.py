#!/usr/bin/env python3
import pytorch_lightning as pl

# Note - you must have torchvision installed for this example
from torchvision import transforms
from os import listdir
from ..csv import load_csv
from pathlib import Path
from ...library.data import load_data_loader


class BaseDataModule(pl.LightningDataModule):

    def __init__(
        self,
        config: dict,
        transform_list: list,
        train_transform_list: list,
        x_label: str,
        y_label: str,
        batch_size: int,
        num_workers: int,
        data_dir: str = './',
    ):
        super().__init__()

        self.data_dir = data_dir
        self.channels = config["channels"]
        self.channel_indexes = config["channel_indexes"]
        self.classes = config["classes"]
        self.id_fields = config["id_fields"]
        self.batch_size = batch_size
        self.num_workers = num_workers

        self._num_channels = len(self.channels)
        chosen_channels = self.channel_indexes
        self.channel_indexes = None
        self.datasets = {}

        if chosen_channels is not None:
            try:
                self.channel_indexes = [
                    self.channels.index(channel_name) for channel_name
                    in chosen_channels
                ]
                self._num_channels = len(self.channel_indexes)
            except ValueError:
                raise Exception(
                    (
                        f"channel indexes {self.channel_indexes} "
                        f"do not match channel names {self.channels}"
                    )
                )

        self.transform = transforms.Compose(transform_list)
        self.train_transform = transforms.Compose(train_transform_list)

        self.x_label = x_label
        self.y_label = y_label

    def prepare_data(self):
        pass

    def load_image(self, dataset):
        raise NotImplementedError(
            "`load_image` hasn't been defined for this data module"
        )

    def get_dims(self, shape):
        raise NotImplementedError(
            "`get_dims` hasn't been defined for this data module"
        )

    def setup(self, stage=None):

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
        val_dataset = self.datasets['valid']
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
        test_dataset = self.datasets['test']
        test_dataloader = load_data_loader(
            test_dataset,
            self.loaders,
            transform=self.transform,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        return test_dataloader
