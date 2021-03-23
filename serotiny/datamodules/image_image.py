from typing import Sequence, Union

import numpy as np
import torch

from torchvision import transforms
import torchio.transforms as tiotransforms
from aicsimageprocessing.resize import resize_to

from ..image import tiff_loader_CZYX
from ..data import load_data_loader
from ..data.loaders import Load3DImage, LoadClass, LoadColumns
from .constants import DatasetFields
from .base_datamodule import BaseDataModule


class ImageImage(BaseDataModule):
    """
    A pytorch lightning datamodule that handles the logic for
    loading 3D ACTK images

    Parameters
    -----------
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

    resize_dims: Sequence[int]
        Resize input images to this size

    encoded_label_suffix: str
        a column of categorical variables is converted into an integer
        representation. This column in named
        encoded_label + encoded_label_suffix
        Example:
            encoded_label = "ChosenMitoticClass"
            encoded_label_suffix = "Integer"

    classes: list
        List of classes in the encoded_label column
    """

    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        x_label: str,
        y_label: str,
        data_dir: str,
        input_channels: Sequence[Union[str, int]],
        output_channels: Sequence[Union[str, int]],
        resize_dims: Sequence[int],
        **kwargs,
    ):
        self.resize_dims = resize_dims
        self.classes = classes

        super().__init__(
            channels=channels,
            select_channels=select_channels,
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
                #tiotransforms.RandomFlip(),  # NOTE: Gui suggestion - we don't want to do random flip for unet
            ],
            x_label=x_label,
            y_label=y_label,
            data_dir=data_dir,
            **kwargs,
        )

        self.x_label = x_label
        self.y_label = y_label

        self.loaders = {
            # Use callable class objects here because lambdas aren't picklable
            "id": LoadColumns(self.id_fields),
            self.x_label: Load3DImage(
                DatasetFields.CellImage3DPath,
                self.num_channels,
                self.input_channels,
                self.transform,
            ),
            self.y_label: Load3DImage(
                DatasetFields.CellImage3DPath,
                self.num_channels,
                self.output_channels,
                self.transform,
            ),
        }

    def load_image(self, dataset):
        """
        Load a single 3D image given a path
        """
        return self.transform(
            tiff_loader_CZYX(
                path_str=dataset[DatasetFields.CellImage3DPath].iloc[0],
                select_channels=self.select_channels,
                output_dtype=np.float32,
            )
        )

    def get_dims(self, img):
        """
        Get dimensions of input image
        """
        return img.shape[1:]

    def train_dataloader(self):
        """
        Instantiate train dataloader.
        """
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
