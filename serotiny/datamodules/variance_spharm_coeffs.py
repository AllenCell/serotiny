from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from typing import Optional
from ..io import load_data_loader
from ..io.loaders import (
    LoadSpharmCoeffs,
    LoadOneHotClass,
    LoadColumns,
    LoadIntClass,
    LoadPCA,
)
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split


class VarianceSpharmCoeffs(pl.LightningDataModule):
    """
    A pytorch lightning datamodule that handles the logic for
    loading the spherical harmonic coefficients

    Parameters
    -----------
    batch_size: int
        Batch size for dataloader

    num_workers: int
        Num of workers for dataloader

    x_label: str
        Column name used to load the input (x)
        Example: "dna_shcoeffs"

    c_label: str
        Column name used to load the input condition (c)
        and make it a one hot encoding.
        Example: "structure_name"

    c_label_ind: str
        Column name used to load the input condition (c)
        as an integer value and not on hot
        NOT related to actual column names in any manifest
        this is just a new name set in this datamodule
        Example: "structure_int"

    id_fields: list
        Column names used to load the Cell ID
        Example: ['CellId']

    x_dim: Size (batch*size) of input data.
        Example: 295

    num_classes: Optional, int
        Example and Default: 25 for structure_name

    data_dir: Optional, str
       Where to save the manifest

    set_zero: Optional, bool
        Whether to set all conditions to 0
        Example and Default: False

    subset: Optional, int
        Whethet to slice train val amd test dataloaders
        Example: 300

    overwrite: Optional, bool
        Whether to overwrite existing file or not
        Default: False

    """

    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        x_label: str,
        c_label: str,
        c_label_ind: str,
        id_fields: list,
        x_dim: int,
        source_path: str,
        modified_source_save_dir: str,
        align: str,
        skew: str,
        num_classes: Optional[int] = None,
        set_zero: Optional[bool] = False,
        subset: Optional[int] = None,
        overwrite: Optional[bool] = False,
        **kwargs,
    ):

        super().__init__()

        self.x_label = x_label
        if not self.x_label:
            self.x_label = "dna_shcoeffs"

        self.c_label = c_label
        if not self.c_label:
            self.c_label = "structure_name"

        self.c_encoded_label = c_label + "_encoded"
        self.id_fields = id_fields

        self.num_classes = num_classes
        if not self.num_classes:
            self.num_classes = 25

        self.source_path = source_path
        self.modified_source_save_dir = modified_source_save_dir

        self.datasets = {}
        self.align = align
        self.skew = skew
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.x_dim = x_dim
        self.c_label_ind = c_label_ind
        self.set_zero = set_zero
        self.subset = subset
        self.overwrite = overwrite
        self.stratify_column = self.c_label

        dfg = pd.read_csv(self.source_path)
        self.dfg = dfg
        self.num_rows = dfg.shape[0]
        self.spharm_cols = [col for col in dfg.columns if self.x_label in col]
        self.spharm_cols = [f for f in self.spharm_cols if "L" in f]

        self.loaders = {
            # Use callable class objects here because lambdas aren't picklable
            "id": LoadColumns(self.id_fields),
            self.c_label: LoadOneHotClass(
                self.num_classes, self.c_encoded_label, self.set_zero
            ),
            self.x_label: LoadSpharmCoeffs(self.x_label, self.spharm_cols),
            self.c_label_ind: LoadIntClass(
                self.num_classes, self.c_encoded_label, self.set_zero
            ),
        }

        if self.c_label in ["DNA_PC", "MEM_PC", "DNA_MEM_PC"]:
            self.loaders[self.c_label] = LoadPCA(self.c_label, set_zero=self.set_zero)
            self.loaders[self.c_label_ind] = LoadPCA(self.c_label, get_inds=True)
            self.stratify_column = "structure_name"

    def setup(self, stage=None):
        """
        Setup train, val and test dataframes. Get image dimensions
        """
        self.modified_source_save_dir = Path(self.modified_source_save_dir)

        all_data = pd.read_csv(
            self.modified_source_save_dir
            / f"variance_rotated_{self.x_label}_{self.c_label}_size_{self.num_rows}_align_{self.align}_skew_{self.skew}.csv"
        )

        if not self.subset:
            self.datasets["test"] = all_data.loc[all_data.split == "test"]
            self.datasets["train"] = all_data.loc[all_data.split == "train"]
            self.datasets["valid"] = all_data.loc[all_data.split == "val"]
        else:
            self.datasets["test"] = (
                all_data.loc[all_data.split == "test"]
                .iloc[: self.subset]
                .reset_index(drop=True)
            )
            self.datasets["train"] = (
                all_data.loc[all_data.split == "train"]
                .iloc[: self.subset]
                .reset_index(drop=True)
            )
            self.datasets["valid"] = (
                all_data.loc[all_data.split == "val"]
                .iloc[: self.subset]
                .reset_index(drop=True)
            )

    def _label_splits(self, row):
        if row.name in self.train_inds:
            return "train"
        elif row.name in self.val_inds:
            return "val"
        elif row.name in self.test_inds:
            return "test"

    def prepare_data(self):
        """
        Prepare dataset
        """
        my_file = (
            Path(self.modified_source_save_dir)
            / f"variance_rotated_{self.x_label}_{self.c_label}_size_{self.num_rows}_align_{self.align}_skew_{self.skew}.csv"
        )
        if my_file.is_file() and self.overwrite is False:
            pass
        else:
            df = self.dfg

            train_inds, test_inds = train_test_split(
                df.index,
                test_size=self.num_rows - int(0.85 * self.num_rows),
                train_size=int(0.85 * self.num_rows),
                stratify=df[self.stratify_column],
                random_state=42,
            )

            dfs = {}
            dfs["train"], dfs["test"] = df.loc[train_inds], df.loc[test_inds]

            train_inds, val_inds = train_test_split(
                dfs["train"].index,
                test_size=self.num_rows - int(0.85 * self.num_rows),
                stratify=dfs["train"][self.stratify_column],
                random_state=42,
            )

            self.train_inds = train_inds
            self.val_inds = val_inds
            self.test_inds = test_inds

            df["split"] = df.apply(lambda row: self._label_splits(row), axis=1)

            # Add class weights
            labels_unique, counts = np.unique(
                df[self.stratify_column], return_counts=True
            )
            class_weights = [sum(counts) / c for c in counts]
            class_weights_dict = {i: j for (i, j) in zip(labels_unique, class_weights)}
            weights = [class_weights_dict[e] for e in df[self.stratify_column]]
            df["ClassWeights"] = weights

            if self.c_label not in ["DNA_PC", "MEM_PC", "DNA_MEM_PC"]:
                # Add label encoding for str names
                df[self.c_label + "_encoded"] = LabelEncoder().fit_transform(
                    df[self.c_label]
                )

                df[self.c_label] = df[self.c_label]
                df[self.c_encoded_label] = df[self.c_label + "_encoded"]

            df.to_csv(
                Path(self.modified_source_save_dir)
                / f"variance_rotated_{self.x_label}_{self.c_label}_size_{self.num_rows}_align_{self.align}_skew_{self.skew}.csv"
            )

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
