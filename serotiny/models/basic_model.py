from typing import Sequence, Union, Optional, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss as Loss

from .base_model import BaseModel

Array = Union[torch.Tensor, np.ndarray, Sequence[float]]


class BasicModel(BaseModel):
    """A minimal Pytorch Lightning wrapper around generic Pytorch models."""

    def __init__(
        self,
        network: nn.Module,
        loss: Loss,
        x_label: str = "x",
        y_label: str = "y",
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        save_predictions: Optional[Callable] = None,
        fields_to_log: Optional[Sequence] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        network: nn.Module
            The network to wrap
        loss: Loss
            The loss function to optimize for
        x_label: str = "x"
            The key used to retrieve the input from dataloader batches
        y_label: str = "y"
            The key used to retrieve the target from dataloader batches
        optimizer: torch.optim.Optimizer = torch.optim.Adam
            The optimizer class
        save_predictions: Optional[Callable] = None
            A function to save the results of `serotiny predict`
        fields_to_log: Optional[Union[Sequence, Dict]] = None
            List of batch fields to store with the outputs. Use a list to log
            the same fields for every training stage (train, val, test, prediction).
            If a list is used, it is assumed to be for test and prediction only
        """
        super().__init__()
        self.network = network
        self.loss = loss

        self._squeeze_y = False

        if save_predictions is not None:
            self.save_predictions = save_predictions
        self.fields_to_log = fields_to_log

    def parse_batch(self, batch):
        return (batch[self.hparams.x_label], batch[self.hparams.y_label])

    def forward(self, x, **kwargs):
        return self.network(x, **kwargs)

    def _step(self, stage, batch, batch_idx, logger):
        x, y = self.parse_batch(batch)

        yhat = self.forward(x)

        if self._squeeze_y:
            loss = self.loss(yhat, y.squeeze())
        else:
            try:
                loss = self.loss(yhat, y)
            except RuntimeError as err:
                if y.shape[-1] == 1:
                    try:
                        loss = self.loss(yhat, y.squeeze())
                        self._squeeze_y = True
                    except Exception as inner_err:
                        raise inner_err from err
                else:
                    raise err

        if stage != "predict":
            self.log(f"{stage}_loss", loss.detach(), logger=logger)

        output = {
            "loss": loss,
            "yhat": yhat.detach().squeeze(),
            "y": y.detach().squeeze(),
        }

        if isinstance(self.fields_to_log, list):
            if stage in ["predict", "test"]:
                for field in self.fields_to_log:
                    output[field] = batch[field]

        elif isinstance(self.fields_to_log, dict):
            if stage in self.fields_to_log:
                for field in self.fields_to_log[stage]:
                    output[field] = batch[field]

        return output
