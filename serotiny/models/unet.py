"""
General conditional beta variational autoencoder module,
implemented as a Pytorch Lightning module
"""
from typing import Sequence
import inspect

import logging

logger = logging.getLogger("lightning")
logger.propagate = False

import torch
import torch.optim as opt
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.utilities.parsing import get_init_args

from ..losses import KLDLoss
from .utils import index_to_onehot, find_optimizer


class UnetModel(pl.LightningModule):
    def __init__(
        self,
        network: nn.Module,
        optimizer: opt.Optimizer,
        loss: nn.Module,
        x_label: str,
        y_label: str,
        input_channels: Sequence[str],
        output_channels: Sequence[str],
        lr: float,
        input_dims: Sequence[int],
    ):
        """
        Instantiate a UnetModel.

        Parameters
        ----------
        network: nn.Module
            Unet network.
        optimizer: opt.Optimizer
            Optimizer for the network
        loss: nn.Module
            Loss function for the recognition loss term
        input_label: str
            Label for the input image, to retrieve it from the batches
        output_label: str
            Label for the output image, to retrieve it from the batches
        in_channels: int
            Number of input channels in the image
            Example: 3
        out_channels: int
            Number of output channels in the image
            Example: 3
        lr: float
            Learning rate for training
        input_dims: Sequence[int]
            Dimensions of the input images
        """
        super().__init__()

        # store init args except for network, to avoid copying
        # large objects unnecessarily
        frame = inspect.currentframe()
        init_args = get_init_args(frame)
        self.save_hyperparameters(*[arg for arg in init_args if arg not in ["network"]])

        self.log_grads = True

        self.network = network
        self.loss = loss

    def parse_batch(self, batch):
        x = batch[self.hparams.x_label].float()
        y = batch[self.hparams.y_label].float()

        return x, y

    def forward(self, x, y):
        #####################
        # train unet
        #####################

        # Forward passes
        y_hat = self.network(x)
        loss = self.loss(y_hat, y)

        return y_hat, loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, y = self.parse_batch(batch)
        y_hat, loss = self(x, y)

        self.manual_backward(loss)
        optimizer = self.hparams.optimizer
        optimizer.step()
        optimizer.zero_grad()

        # Default logger=False for training_step
        # set it to true to log train loss to all lggers
        self.log("train loss", loss, logger=True)
        return {"loss": loss, "batch_idx": batch_idx}

    def validation_step(self, batch, batch_idx):
        x, y = self.parse_batch(batch)
        y_hat, loss = self(x, y)

        # Default logger=False for training_step
        # set it to true to log train loss to all lggers
        self.log("validation loss", loss, logger=True)

        return {
            "validation_loss": loss,
            "y_hat": y_hat,
            "batch_idx": batch_idx,
        }

    def test_step(self, batch, batch_idx):
        x, y = self.parse_batch(batch)
        y_hat, loss = self(x, y)

        return {
            "test_loss": loss,
            "y_hat": y_hat,
            "batch_idx": batch_idx,
        }

    def test_step_end(self, output):
        # TODO: this should be using a callback defined elsewhere to decide
        # what to do after the test step. Sometimes we want metrics, sometimes
        # we want to store the latent representations, etc.
        loss = output["loss"]
        self.log("test loss", loss, logger=True)

    def configure_optimizers(self):
        opt_class = find_optimizer(self.hparams.optimizer)
        optimizer = opt_class(self.network.parameters(), self.hparams.lr)

        return (
            {
                "optimizer": optimizer,
                "monitor": "val_accuracy",
                "interval": "epoch",
                "frequency": 1,
                "strict": True,
            },
        )

    def on_after_backward(self):
        # example to inspect gradient information in tensorboard
        if (
            self.log_grads and self.trainer.global_step % 100 == 0
        ):  # don't make the file huge
            params = self.state_dict()
            for k, v in params.items():
                grads = v
                name = k
                # logger[0] is tensorboard logger
                self.logger[0].experiment.add_histogram(
                    tag=name, values=grads, global_step=self.trainer.global_step
                )
