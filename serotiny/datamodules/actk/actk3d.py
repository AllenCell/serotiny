# Note - you must have torchvision installed for this example
from torchvision import transforms
from ..image import tiff_loader_CZYX
import numpy as np
from ...constants import DatasetFields
from ...library.data import load_data_loader, Load3DImage, LoadClass, LoadId
from .base_datamodule import BaseDataModule
import torch

from aicsimageprocessing.resize import resize_to
from .utils import subset_channels

import torchio.transforms as tiotransforms


class ACTK3DDataModule(BaseDataModule):
    def __init__(
        self,
        config: dict,
        batch_size: int,
        num_workers: int,
        x_label: str,
        y_label: str,
        data_dir: str,
        resize_dims=(64, 128, 96),
    ):

        self.channels = config["channels"]
        self.select_channels = config["select_channels"]
        self.classes = config["classes"]
        self.num_channels = len(self.select_channels)
        self.resize_dims = resize_dims

        super().__init__(
            config=config,
            batch_size=batch_size,
            num_workers=num_workers,
            transform_list=[
                transforms.Lambda(
                    lambda x: resize_to(x, (self.num_channels, *resize_dims))
                ),
                transforms.Lambda(lambda x: torch.tensor(x)),
            ],
            train_transform_list=[
                transforms.Lambda(
                    lambda x: resize_to(x, (self.num_channels, *resize_dims))
                ),
                transforms.Lambda(lambda x: torch.tensor(x)),
                tiotransforms.ToCanonical(),
                tiotransforms.RandomFlip(),
            ],
            x_label=x_label,
            y_label=y_label,
            data_dir=data_dir,
        )

        self.x_label = x_label
        self.y_label = y_label
        self.y_encoded_label = y_label + "Integer"

        self.loaders = {
            # Use callable class objects here because lambdas aren't picklable
            "id": LoadId(self.id_fields),
            self.y_label: LoadClass(len(self.classes), self.y_encoded_label),
            self.x_label: Load3DImage(
                DatasetFields.CellImage3DPath,
                self.num_channels,
                self.select_channels,
                self.transform,
            ),
        }

    def load_image(self, dataset):
        return self.transform(
            tiff_loader_CZYX(
                path_str=dataset[DatasetFields.CellImage3DPath].iloc[0],
                select_channels=self.select_channels,
                output_dtype=np.float32,
            )
        )

    def get_dims(self, img):
        return img.shape[1:]

    def train_dataloader(self):
        train_dataset = self.datasets["train"]
        train_loaders = self.loaders.copy()
        train_loaders[self.x_label] = Load3DImage(
            DatasetFields.CellImage3DPath,
            self.num_channels,
            self.select_channels,
            self.train_transform,
        )
        train_dataloader = load_data_loader(
            train_dataset,
            train_loaders,
            transform=self.train_transform,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        return train_dataloader
