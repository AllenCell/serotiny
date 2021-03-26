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
        auto_padding: bool = False,
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
        x_label: str
            Label for the input image, to retrieve it from the batches
        y_label: str
            Label for the output image, to retrieve it from the batches
        input_channels: Sequence[int]
            Which channel (indices) to use as input
        output_channels: Sequence[int]
            Which channel (indices) to use as output
        lr: float
            Learning rate for training
        input_dims: Sequence[int]
            Dimensions of the input images
        auto_padding: bool = False
            Whether to apply padding to ensure images from down double conv match up double conv
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
        
        self.input_dims = input_dims

    def parse_batch(self, batch):
        # TODO: The values of x_label and y_label are used as keys in the batch dictionary.
        #       If they are the same column names, there may be conflict
        #print(f'batch = {batch}')
        x = batch[self.hparams.x_label].float()
        y = batch[self.hparams.y_label].float()

        return x, y

    # Calculate the amount of padding required given an input image dimension,
    # a specific Unet depth and a channel_fan factor
    
    # Issues:
    #   - Padding may introduce a lot of blank space depending on the input
    #     image size and the network depth
    #   - May not be most efficient to do it in the forward pass since the
    #     amount of padding is the same, so we should not need to recalculate
    #     this for each image
    
    def get_unet_padding(self, input_dims, depth, channel_fan):

        input_paddings = []

        # Loop through each dimension
        for input_size in input_dims:

            next_size = input_size

            # For each dimension, traverse down the network
            for current_depth in range(depth, -1, -1):

                orig_size = next_size

                # If we are not at the bottom, check to see if we need to pad
                if current_depth > 0:
                    # Pad if we cannot divide the image size by channel_fan to
                    # get an integer
                    if orig_size // channel_fan != orig_size / channel_fan:
                        next_size_float = int(orig_size / channel_fan)
                        new_size = channel_fan * (next_size_float + 1)
                    else:
                        new_size = orig_size
                        
                    # Calculate the image size of the next level down
                    next_size = new_size // channel_fan
                    
                else:
                    new_size = orig_size
                    next_size = orig_size

                print(f'{current_depth}: orig_size = {orig_size}, new_size = {new_size}, next_size = {next_size}')

            # Calculate the new input image size given the padding, and save the padding
            # for all the dimensions in a format that is taken by F.pad() (which goes in the
            # reverse order as the input dimensions)
            new_input_size = new_size * (channel_fan**depth)
            padding = new_input_size - input_size
            padding_left = padding // 2
            padding_right = padding - padding_left

            print(f'input_size = {input_size}, new_input_size = {new_input_size}, padding = {padding_left, padding_right}')

            input_paddings.insert(0, padding_right)
            input_paddings.insert(0, padding_left)

        return tuple(input_paddings)
    
    def forward(self, x, y):
        #####################
        # train unet
        #####################

        if self.hparams.auto_padding:
            #print(f'self.network.depth = {self.network.depth}')
            #print(f'self.network.channel_fan = {self.network.channel_fan}')
            #print(f'self.input_dims = {self.input_dims}')
            
            # Ignore the first dimension (number of channels)
            input_dims_xyz = self.input_dims[1:]
            #print(f'input_dims_xyz = {input_dims_xyz}')
            
            padding = self.get_unet_padding(list(input_dims_xyz), self.network.depth, self.network.channel_fan)
            #print(f'padding = {padding}')
            
            # We need to pad both x and y, otherwise, they will have different sizes.
            # This assumes that x and y have the same sizes to begin with
            x = F.pad(x, padding, mode="constant", value=0)
            y = F.pad(y, padding, mode="constant", value=0)
            
        # Forward passes
        y_hat = self.network(x)
        loss = self.loss(y_hat, y)

        return y_hat, loss

    # NOTE: Had to remove optimizer_idx if using automatic backprop
    #def training_step(self, batch, batch_idx, optimizer_idx):
    def training_step(self, batch, batch_idx):
        print(f'In training step (batch_idx = {batch_idx})')
        x, y = self.parse_batch(batch)
        y_hat, loss = self(x, y)

        # NOTE: Nothing needs to be done here with backprop if using
        #       automatic optimization
        
        #optimizer = self.hparams.optimizer
        
        #opt = self.optimizers()
        #self.manual_backward(loss, opt)
        #optimizer.step()
        #optimizer.zero_grad()
        #opt.step()
        #opt.zero_grad()

        # Default logger=False for training_step
        # set it to true to log train loss to all lggers
        self.log("train loss", loss, logger=True)
        return {"loss": loss, "batch_idx": batch_idx}

    def validation_step(self, batch, batch_idx):
        print(f'In validation step (batch_idx = {batch_idx})')
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
        print(f'In test step (batch_idx = {batch_idx})')
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
        loss = output["test_loss"]
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
