"""
General regression module, implemented as a Pytorch Lightning module
"""

from typing import Union, Dict

import torch.nn as nn
from torch.nn.modules.loss import _Loss as Loss
from .base_graph import BaseGraph


class GraphNodePredictionModel2(BaseGraph):
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
        network: Union[nn.Module, Dict],
        lr: float,
        optimizer: str,
        task: str,
        dataloader_type: str,
        loss: Union[Loss, Dict] = nn.MSELoss(),
    ):
        self.dataloader_type = dataloader_type
        super().__init__(
            network=network, lr=lr, optimizer=optimizer, task=task, loss=loss,
        )

    def training_step(self, batch, batch_idx):

        if self.dataloader_type == "graph_saint":
            self.network.set_aggr("add")
        else:
            try:
                self.network.set_aggr("mean")
            except:
                pass

        x, edge_index, edge_weight, _, y = self.parse_batch(batch)

        if self.dataloader_type == "graph_saint":
            edge_weight = batch.edge_norm * edge_weight
            loss, yhat, y = self.base_step(
                x, y, edge_index, edge_weight, batch.mask, batch.node_norm,
            )
        else:
            edge_weight = batch.edge_weight
            loss, yhat, y = self.base_step(
                x, y, edge_index, edge_weight, batch.mask, None
            )

        self.log("train_loss", loss.detach(), logger=True)

        return {
            "loss": loss,
            "train_loss": loss.detach(),
            "train_mask": batch.mask,
            "yhat": yhat.detach(),
            "target": y,
            "batch_idx": batch_idx,
        }

    def validation_step(self, batch, batch_idx):
        try:
            self.network.set_aggr("mean")
        except:
            pass

        x, edge_index, _, _, y = self.parse_batch(batch)

        loss, yhat, y = self.base_step(
            x, y, edge_index, batch.edge_weight, batch.mask, None
        )

        self.log("val_loss", loss.detach(), logger=True)

        return {
            "val_loss": loss,
            "val_mask": batch.mask,
            "yhat": yhat,
            "target": y,
            "batch_idx": batch_idx,
        }

    def test_step(self, batch, batch_idx):
        try:
            self.network.set_aggr("mean")
        except:
            pass

        x, edge_index, _, _, y = self.parse_batch(batch)

        loss, yhat, y = self.base_step(
            x, y, edge_index, batch.edge_weight, batch.mask, None
        )

        self.log("test_loss", loss.detach(), logger=True)

        return {
            "test_loss": loss,
            "test_mask": batch.mask,
            "yhat": yhat,
            "target": y,
            "batch_idx": batch_idx,
        }
