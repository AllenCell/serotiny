from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from ..data import load_data_loader
from ..data.loaders import LoadSpharmCoeffs, LoadOneHotClass, LoadColumns
import pytorch_lightning as pl


class VarianceSpharmCoeffs(pl.LightningDataModule):
    """
    A pytorch lightning datamodule that handles the logic for
    loading the spherical harmonic coefficients

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

    id_fields: List[str]
        Id column name for loader

    """

    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        x_label: str,
        c_label: str,
        id_fields: list,
        **kwargs
    ):

        super().__init__()

        self.x_label = x_label
        self.c_label = c_label
        self.c_encoded_label = c_label + "_encoded"
        self.id_fields = id_fields
        self.num_classes = 25
        self.data_dir = "/allen/aics/modeling/ritvik/projects/serotiny/"
        self.datasets = {}
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.loaders = {
            # Use callable class objects here because lambdas aren't picklable
            "id": LoadColumns(self.id_fields),
            self.c_label: LoadOneHotClass(self.num_classes, self.c_encoded_label),
            self.x_label: LoadSpharmCoeffs(),
        }

        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        self.dims = (28, 28)

    def setup(self, stage=None):
        """
        Setup train, val and test dataframes. Get image dimensions
        """
        self.data_dir = Path(self.data_dir)

        all_data = pd.read_csv(self.data_dir / "variance_spharm_coeffs.csv")

        all_data["structure_name_encoded"] = LabelEncoder().fit_transform(
            all_data["structure_name"]
        )

        all_data[self.c_label] = all_data["structure_name"]
        all_data[self.c_encoded_label] = all_data["structure_name_encoded"]

        self.datasets["test"] = all_data.loc[all_data.split == "test"]
        self.datasets["train"] = all_data.loc[all_data.split == "train"]
        self.datasets["valid"] = all_data.loc[all_data.split == "val"]

    def prepare_data(self):
        """
        Download dataset
        """
        pass

    def train_dataloader(self):
        """
        Instantiate train dataloader.
        """
        train_dataset = self.datasets["train"]
        train_dataloader = load_data_loader(
            train_dataset,
            self.loaders,
            transform=None,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            weights_col="ClassWeights",
        )

        return train_dataloader

    def val_dataloader(self):
        """
        Instantiate val dataloader. This should ideally be implemented
        in base_datamodule
        """
        val_dataset = self.datasets["valid"]
        val_dataloader = load_data_loader(
            val_dataset,
            self.loaders,
            transform=None,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            weights_col="ClassWeights",
        )

        return val_dataloader

    def test_dataloader(self):
        """
        Instantiate test dataloader. This should ideally be implemented
        in base_datamodule
        """
        test_dataset = self.datasets["test"]
        test_dataloader = load_data_loader(
            test_dataset,
            self.loaders,
            transform=None,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            weights_col="ClassWeights",
        )

        return test_dataloader
