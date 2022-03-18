import pytorch_lightning as pl
from typing import Union
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np
import multiprocessing as mp
import pandas as pd
from serotiny.io.dataframe import filter_columns
from serotiny.datamodules.datasets import TrajectoryDataset, TrajectoriesDataset
from random import sample


class TrajectoryDataModule2(pl.LightningDataModule):
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

        df_val = pd.read_csv(manifest)

        min_t = 500
        all_df = []
        for track, df1 in df_val.groupby("track_id"):
            # this_df1 = (
            #     df1.groupby("digitized_normalized_time_int")
            #     .sample(n=13)  # try 13
            #     .reset_index(drop=True)
            # )
            this_df1 = df1.reset_index(drop=True)
            all_df.append(this_df1)

            if df1.shape[0] < min_t:
                min_t = df1.shape[0]

        all_df = pd.concat(all_df, axis=0)
        # import ipdb
        # ipdb.set_trace()
        cols = filter_columns(all_df.columns, **cols_to_filter)
        df2 = all_df[cols].copy()
        df1 = df2.loc[:, (df2 != 0).all()]
        cols = list(df1.columns)

        self.cols = cols

        if normalize:
            all_df[cols] = all_df[cols].apply(
                lambda x: (x - x.min()) / (x.max() - x.min())
            )

        all_tracks = len(all_df["track_id"].unique())
        train_tracks = int(0.6 * all_tracks)
        val_tracks = int(0.2 * all_tracks)

        list_all_tracks = list(all_df["track_id"].unique())
        train_tracks = sample(list_all_tracks, train_tracks)
        not_train_tracks = list(
            set(list_all_tracks).symmetric_difference(set(train_tracks))
        )
        val_tracks = sample(
            not_train_tracks,
            val_tracks,
        )
        train_val = train_tracks + val_tracks
        test_tracks = list(set(list_all_tracks).symmetric_difference(set(train_val)))

        all_traj = []
        for track, df1 in all_df.groupby("track_id"):
            if track in train_tracks:
                this_traj = TrajectoryDataset(
                    lagtime,
                    np.array(df1[cols]).astype(np.float32),
                    np.array(df1["CellId"]),
                    x_label,
                    xhat_label,
                )
                all_traj.append(this_traj)

        dataset_train = TrajectoriesDataset(all_traj)

        all_traj = []
        for track, df1 in all_df.groupby("track_id"):
            if track in val_tracks:
                this_traj = TrajectoryDataset(
                    lagtime,
                    np.array(df1[cols]).astype(np.float32),
                    np.array(df1["CellId"]),
                    x_label,
                    xhat_label,
                )
                all_traj.append(this_traj)

        dataset_val = TrajectoriesDataset(all_traj)

        all_traj = []
        for track, df1 in all_df.groupby("track_id"):
            if track in test_tracks:
                this_traj = TrajectoryDataset(
                    lagtime,
                    np.array(df1[cols]).astype(np.float32),
                    np.array(df1["CellId"]),
                    x_label,
                    xhat_label,
                )
                all_traj.append(this_traj)

        dataset_test = TrajectoriesDataset(all_traj)

        self.datasets = {}

        self.datasets["train"] = dataset_train
        self.datasets["validation"] = dataset_val
        self.datasets["test"] = dataset_test

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
