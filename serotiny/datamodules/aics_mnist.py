from typing import Sequence, Any

import glob
import tarfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from torchvision import transforms

from ..io import png_loader
from ..io import download_quilt_data
from ..io import load_data_loader
from ..io.loaders import Load2DImage, LoadClass, LoadColumns
from .base_datamodule import BaseDataModule


def make_manifest(dataset_path):
    """
    Make a manifest from dataset path
    """
    cells = []
    for split in ["train", "test"]:
        _split_path = str((Path(dataset_path) / split) / "*")
        for structure_path in glob.glob(_split_path):
            _struct_path = str(Path(structure_path) / "*")
            structure = structure_path.split("/")[-1]
            for cell_img in glob.glob(_struct_path):
                cells.append(
                    dict(
                        cellpath2dict(cell_img),
                        structure=structure,
                        split=split,
                        path=str(Path(cell_img).resolve()),
                    )
                )

    return pd.DataFrame(cells)


def cellpath2dict(path):
    """
    Convert a given cell path to a dict
    """
    cell = path.split("/")[-1]
    cell = cell.split(".")[0]
    cell = cell.split("_")
    return {cell[i * 2]: cell[i * 2 + 1] for i in range(len(cell) // 2)}


class AICS_MNIST_DataModule(BaseDataModule):
    """
    A pytorch lightning datamodule that handles the logic for
    loading the AICS MNIST dataset

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

    id_fields: Sequence
        Id column name for loader

    channels: Sequence
        List of channels in the images

    select_channels: Sequence
        List of channels to subset the original channel list

    data_dir: str
        Path to data folder containing csv's for train, val,
        and test splits

    """

    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        data_dir: str,
        x_label: str,
        y_label: str,
        channels: Sequence[Any],
        select_channels: Sequence[Any],
        id_fields: Sequence[Any],
        **kwargs
    ):

        super().__init__(
            channels=channels,
            select_channels=select_channels,
            batch_size=batch_size,
            num_workers=num_workers,
            transform_list=[],
            train_transform_list=[
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ],
            x_label=x_label,
            y_label=y_label,
            data_dir=data_dir,
            **kwargs
        )

        self.y_encoded_label = y_label + "_encoded"
        self.id_fields = id_fields
        self.num_classes = 10

        self.loaders = {
            # Use callable class objects here because lambdas aren't picklable
            "id": LoadColumns(self.id_fields),
            self.y_label: LoadClass(self.num_classes, self.y_encoded_label),
            self.x_label: Load2DImage(
                "path",
                self.num_channels,
                self.channel_indexes,
                self.transform,
            ),
        }

        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        self.dims = (28, 28)

    @classmethod
    def get_dataset(cls, data_dir):
        data_dir = Path(data_dir)

        if not (data_dir / "aics_mnist_rgb.tar.gz").exists():
            download_quilt_data(
                package="aics/aics_mnist",
                bucket="s3://allencell",
                data_save_loc=data_dir,
            )

        if not (data_dir / "aics_mnist_rgb").exists():
            with tarfile.open(data_dir / "aics_mnist_rgb.tar.gz") as f:
                f.extractall(data_dir)

        if not (data_dir / "aics_mnist_rgb.csv").exists():
            manifest = make_manifest(data_dir / "aics_mnist_rgb")
            manifest["structure_encoded"] = LabelEncoder().fit_transform(
                manifest["structure"]
            )
            manifest.to_csv(data_dir / "aics_mnist_rgb.csv", index=False)

    def setup(self, stage=None):
        """
        Setup train, val and test dataframes. Get image dimensions
        """
        self.data_dir = Path(self.data_dir)

        all_data = pd.read_csv(self.data_dir / "aics_mnist_rgb.csv")

        all_data[self.y_label] = all_data["structure"]
        all_data[self.y_encoded_label] = all_data["structure_encoded"]

        self.datasets["test"] = all_data.loc[all_data.split == "test"]

        train_val = all_data.loc[all_data.split == "train"]

        train_inds, val_inds = train_test_split(
            train_val.index,
            test_size=len(self.datasets["test"]),
            stratify=train_val.structure,
            random_state=42,
        )

        self.datasets["train"] = train_val.loc[train_inds]
        self.datasets["valid"] = train_val.loc[val_inds]
        self.datasets["valid"]["split"] = "valid"

        self.dims = (28, 28)

    def prepare_data(self):
        """
        Download dataset
        """
        self.get_dataset(self.data_dir)

    def load_image(self, dataset):
        """
        Load a single 2D image given a path
        """
        return png_loader(
            dataset["path"].iloc[0],
            channel_order="CYX",
            indexes={"C": self.channel_indexes or range(self.num_channels)},
            transform=self.transform,
        )

    def get_dims(self, img):
        """
        Get dimensions of input image
        """
        return (img.shape[1], img.shape[2])

    def train_dataloader(self):
        """
        Instantiate train dataloader.
        """
        train_dataset = self.datasets["train"]
        train_loaders = self.loaders.copy()
        train_loaders[self.x_label] = Load2DImage(
            "path",
            self.num_channels,
            self.channel_indexes,
            self.train_transform,
        )
        train_dataloader = load_data_loader(
            train_dataset,
            train_loaders,
            transform=self.train_transform,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            weights_col=None,
        )

        return train_dataloader

    def val_dataloader(self):
        """
        Instantiate val dataloader. This should ideally be implemented
        in base_datamodule
        """
        val_dataset = self.datasets["valid"]
        val_loaders = self.loaders.copy()
        val_loaders[self.x_label] = Load2DImage(
            "path",
            self.num_channels,
            self.channel_indexes,
            self.transform,
        )
        val_dataloader = load_data_loader(
            val_dataset,
            val_loaders,
            transform=self.transform,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            weights_col=None,
        )

        return val_dataloader

    def test_dataloader(self):
        """
        Instantiate test dataloader. This should ideally be implemented
        in base_datamodule
        """
        test_dataset = self.datasets["test"]
        test_loaders = self.loaders.copy()
        test_loaders[self.x_label] = Load2DImage(
            "path",
            self.num_channels,
            self.channel_indexes,
            self.transform,
        )
        test_dataloader = load_data_loader(
            test_dataset,
            test_loaders,
            transform=self.transform,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            weights_col=None,
        )

        return test_dataloader
