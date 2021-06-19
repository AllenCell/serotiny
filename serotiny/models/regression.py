"""
General regression module, implemented as a Pytorch Lightning module
"""

from typing import Optional, Union, Dict
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss as Loss

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from serotiny.utils import get_class_from_path
from serotiny.models._utils import find_optimizer


class RegressionModel(pl.LightningModule):
    """
    Pytorch lightning module that implements regression model logic

    Parameters
    -----------
    network:
        Neural network for training a regression problem

    x_label: str
        Key for loading input image
        Example: "projection_image"

    y_label: str
        Key for loading image label

    in_channels: int
        Number of input channels in the image
        Example: 3

    lr: float
        Learning rate for the optimizer. Example: 1e-3

    optimizer: str
        str key to instantiate optimizer. Example: "Adam"

    lr_scheduler: str
        str key to instantiate learning rate scheduler

    """

    def __init__(
        self,
        network: Union[nn.Module, str],
        x_label: str,
        y_label: str,
        lr: float,
        optimizer: str,
        loss: Union[Loss, str] = nn.MSELoss,
        network_config: Optional[Dict] = None,
    ):
        super().__init__()
        # Can be accessed via checkpoint['hyper_parameters']
        self.save_hyperparameters()
        if isinstance(loss, str):
            loss = get_class_from_path(loss)
        if isinstance(network, str):
            network = get_class_from_path(network)
            network = network(**network_config)


        self.loss = loss()
        self.network = network

    def parse_batch(self, batch):
        """
        Retrieve x and y from batch
        """
        x = batch[self.hparams.x_label]
        y = batch[self.hparams.y_label]

        return x.float(), y.float()

    def forward(self, x):
        return self.network(x)

    def _step(self, stage, batch, batch_idx, logger):
        x, y = self.parse_batch(batch)

        yhat = self.network(x)

        loss = self.loss(y, yhat)
        self.log(f"{stage} loss", loss, logger=logger)

        results = {
            "loss": loss,
            "yhat": yhat.detach(),
            f"{stage}_loss": loss.detach(),  # for epoch end logging purposes
            "batch_idx": batch_idx,
        }

        return results

    def training_step(self, batch, batch_idx):
        return self._step("train", batch, batch_idx, logger=True)

    def validation_step(self, batch, batch_idx):
        return self._step("val", batch, batch_idx, logger=True)

    def test_step(self, batch, batch_idx):
        return self._step("test", batch, batch_idx, logger=False)

    def configure_optimizers(self):
        optimizer_class = find_optimizer(self.hparams.optimizer)
        optimizer = optimizer_class(self.parameters(), lr=self.hparams.lr)

        return (
            {
                "optimizer": optimizer,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
                "strict": True,
            },
        )

