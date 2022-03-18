"""General regression module, implemented as a Pytorch Lightning module."""

from typing import Union, Dict

import torch.nn as nn
from torch.nn.modules.loss import _Loss as Loss
import pytorch_lightning as pl

from serotiny.utils import init
from serotiny.models._utils import find_optimizer
from serotiny.networks.weight_init import weight_init


class BaseGraph(pl.LightningModule):
    """Pytorch lightning module that implements regression model logic.

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
        loss: Union[Loss, Dict] = nn.MSELoss(),
    ):
        super().__init__()
        # Can be accessed via checkpoint['hyper_parameters']
        # self.save_hyperparameters()
        if isinstance(loss, dict):
            self.loss = init(loss)
        else:
            self.loss = loss

        if isinstance(network, dict):
            network = init(network)

        self.network = network.apply(weight_init)
        self.lr = lr
        self.optimizer = optimizer
        self.task = task
        self._cache = dict()

    def parse_batch(self, batch):
        """Retrieve x and y from batch."""
        x = batch.x
        if self.task == "classification":
            y = batch.y.long()
        elif self.task == "regression":
            y = batch.y.float()
            y = y.unsqueeze(dim=1)
        else:
            y = None

        edge_index = batch.edge_index
        edge_weight = batch.edge_weight
        edge_attr = batch.edge_attr

        return x.float(), edge_index, edge_weight, edge_attr, y

    def forward(self, x):
        return self.network(x)

    def base_step(self, x, y, edge_index, edge_weight, mask, node_norm=None):

        yhat = self.network(x, edge_index, edge_weight)

        loss = self.loss(yhat, y)
        y = y[mask]
        yhat = yhat[mask]

        # Node norm requires graph saint random walk
        if node_norm is not None:
            loss = (loss * node_norm)[mask].sum()
        else:
            loss = loss[mask].sum()

        if self.task == "regression":
            loss = loss.squeeze()

        return loss, yhat, y

    def configure_optimizers(self):
        optimizer_class = find_optimizer(self.optimizer)
        optimizer = optimizer_class(self.parameters(), lr=self.lr)

        return (
            {
                "optimizer": optimizer,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
                "strict": True,
            },
        )
