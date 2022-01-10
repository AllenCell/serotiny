import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import numpy as np
import multiprocessing as mp
from serotiny.datamodules.datasets import TrajectoryDataset, TrajectoriesDataset
from deeptime.data import sqrt_model, swissroll_model


class SyntheticTrajectoryDataModule(pl.LightningDataModule):
    """
    A pytorch lightning datamodule that handles the logic for
    loading a Gaussian dataset

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
        xhat_label: str,
        lagtime: int,
        pin_memory: bool = True,
        drop_last: bool = False,
        **kwargs
    ):

        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.x_label = x_label
        self.xhat_label = xhat_label
        self.lagtime = lagtime
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        all_traj = []
        all_dtraj = []

        for _ in range(1):
            dtraj, traj = sqrt_model(n_samples=10000)
            # dtraj, traj = swissroll_model(n_samples=10000)
            this_traj = TrajectoryDataset(
                lagtime,
                traj.astype(np.float32),
                dtraj.astype(np.float32),
                None,
                x_label,
                xhat_label,
            )
            all_traj.append(this_traj)
            all_dtraj.append(dtraj)

        dataset = TrajectoriesDataset(all_traj)
        self.dataset = dataset
        self.dtraj = all_dtraj

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
            shuffle = True
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
