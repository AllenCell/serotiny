import pytorch_lightning as pl
from typing import Union
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np
import multiprocessing as mp
import pandas as pd
from serotiny.io.dataframe import filter_columns
from serotiny.datamodules.datasets import TrajectoryDataset, TrajectoriesDataset


class TrajectoryDataModule(pl.LightningDataModule):
    """A pytorch lightning datamodule that handles the logic for loading a Gaussian
    dataset.

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
        manifest: Union[Path, str],
        batch_size: int,
        num_workers: int,
        x_label: str,
        xhat_label: str,
        cols_to_filter: dict,
        lagtime: int,
        pin_memory: bool = True,
        drop_last: bool = False,
        normalize: bool = False,
        filter_nonzero: bool = False,
        condition_label: str = None,
        condition: bool = None,
        condition_shuffle: bool = False,
        sample_equal_timepoints: bool = False,
        cols_to_normalize: list = None,
        **kwargs
    ):

        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.x_label = x_label
        self.xhat_label = xhat_label
        self.cols_to_filter = cols_to_filter
        self.lagtime = lagtime
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.filter_nonzero = filter_nonzero
        self.condition_label = condition_label
        self.condition = condition
        self.condition_shuffle = condition_shuffle
        self.sample_equal_timepoints = sample_equal_timepoints
        self.cols_to_normalize = cols_to_normalize

        df_val = pd.read_csv(manifest)

        min_t = 500
        all_df = []
        for track, df1 in df_val.groupby("track_id"):
            if df1.shape[0] > 30:
                this_df1 = df1.sort_values(by="T_index").reset_index(drop=True)
                all_df.append(this_df1)
                # print(this_df1.shape)
                if this_df1.shape[0] < min_t:
                    min_t = this_df1.shape[0]
        # import ipdb
        # ipdb.set_trace()
        if self.sample_equal_timepoints:
            all_df = []
            for track, df1 in df_val.groupby("track_id"):
                if df1.shape[0] > 30:
                    this_df1 = (
                        df1.sample(n=min_t)
                        .sort_values(by="T_index")
                        .reset_index(drop=True)
                    )
                    all_df.append(this_df1)
                    # import ipdb
                    # ipdb.set_trace()

        all_df = pd.concat(all_df, axis=0)
        # import ipdb
        # ipdb.set_trace()
        cols = filter_columns(all_df.columns, **cols_to_filter)

        # Add for spherical harmonics
        if self.filter_nonzero:
            df2 = all_df[cols].copy()
            df1 = df2.loc[:, (df2 != 0).all()]
            cols = list(df1.columns)

        self.cols = cols

        if normalize:
            if self.cols_to_normalize is not None:
                all_df[self.cols_to_normalize] = all_df[self.cols_to_normalize].apply(
                    lambda x: (x - x.min()) / (x.max() - x.min())
                )
                print("self cols norm")
            else:
                all_df[cols] = all_df[cols].apply(
                    lambda x: (x - x.min()) / (x.max() - x.min())
                )
        if self.condition_label is not None:
            if isinstance(condition_label, str):
                condition_label = [condition_label]
                all_df[condition_label] = all_df[condition_label].apply(
                    lambda x: (x - x.min()) / (x.max() - x.min())
                )

        all_traj = []
        for track, df1 in all_df.groupby("track_id"):
            if self.condition_label is not None:
                condition_values = np.array(df1[condition_label]).astype(np.float32)
                if self.condition_shuffle:
                    np.random.shuffle(condition_values)

            else:
                condition_values = None

            this_traj = TrajectoryDataset(
                lagtime,
                np.array(df1[cols]).astype(np.float32),
                np.array(df1["CellId"]),
                condition_values,
                x_label,
                xhat_label,
            )
            all_traj.append(this_traj)

        dataset = TrajectoriesDataset(all_traj)
        self.dataset = dataset

        train_size = int(len(dataset) * 0.6)
        val_size = int((len(dataset) - int(len(dataset) * 0.6)) / 2)
        test_size = len(dataset) - train_size - val_size

        train_data, val_data, test_data = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )

        self.datasets = {}

        self.datasets["train"] = train_data
        self.datasets["validation"] = val_data
        self.datasets["test"] = test_data

    def make_dataloader(self, mode):
        if mode != "train":
            batch_size = len(self.datasets[mode])
            shuffle = False
        else:
            batch_size = self.batch_size
            shuffle = True

        return DataLoader(
            dataset=self.datasets[mode],
            batch_size=batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            shuffle=shuffle,
            multiprocessing_context=mp.get_context("fork"),
        )

    def train_dataloader(self):
        return self.make_dataloader("train")

    def val_dataloader(self):
        return self.make_dataloader("validation")

    def test_dataloader(self):
        return self.make_dataloader("test")
