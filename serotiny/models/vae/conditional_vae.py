"""
Base conditional variational autoencoder module, implemented as a
Pytorch Lightning module
"""

from typing import Sequence, Union, Optional, Dict
import inspect

import logging
logger = logging.getLogger("lightning")
logger.propagate = False

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss as Loss

from .base_vae import BaseVAE

Array = Union[torch.Tensor, np.array]

class ConditionalVAE(BaseVAE):
    def __init__(
        self,
        encoder: Union[nn.Module, Dict],
        decoder: Union[nn.Module, Dict],
        latent_dim: Union[int, Sequence[int]],
        optimizer: str,
        lr: float,
        beta: float,
        x_label: str,
        c_label: Union[str, int, Sequence[int]],
        recon_loss: Union[Loss, str] = torch.nn.MSELoss,
        condition_mode: str = "label",
        prior_mode: str = "isotropic",
        prior_logvar: Optional[Array] = None,
        learn_prior_logvar: bool = False,
    ):
        """
        Instantiate a conditional VAE model

        Parameters
        ----------
        encoder: nn.Module
            Encoder network.
        decoder: nn.Module
            Decoder network.
        optimizer: opt.Optimizer
            Optimizer to use
        recon_loss: Callable
            Loss function for the reconstruction loss term
        lr: float
            Learning rate for training
        beta: float
            Beta parameter - the weight of the KLD term in the loss function
        x_label: str
            String label used to retrieve X from the batches
        c_label: str
            String label used to retrieve conditioning input from the batches
        condition_mode: str
            String describing the type of conditioning to use.
            If `condition_mode` is "channel",
            it expects c_label to be an integers or list of integers, and the input
            x to be at least 2D. In that case, c_label selects the channel or
            channels to be used as conditioning input.
            If `condition_mode` is "label", it expects c_label to be a string,
            used to retrieve the conditioning input from a given batch
        prior_mode: str
            String describing which type of prior to use. (Only "isotropic" and
            "anisotropic" currently supported)
        prior_logvar: Optional[Array]
            Array of values to be used as either the fixed value or initialization
            value for the diagonal of the prior covariance matrix
        learn_prior_logvar: bool
            Boolean flag to determine whether to learn the prior log-variances
        """

        super().__init__(encoder, decoder, latent_dim, optimizer, lr,
                         beta, x_label, recon_loss,
                         prior_mode, prior_logvar, learn_prior_logvar)

        if condition_mode not in ("channel", "label"):
            raise ValueError("`condition_mode` should be "
                             "either 'channel' or 'label")

        self.c_label = c_label
        if condition_mode == "channel":
            if isinstance(c_label, int):
                self.c_label = [c_label]

            assert isinstance(self.c_label, Sequence[int])

        self.condition_mode = condition_mode

    def parse_batch(self, batch):
        if self.condition_mode == "label":
            x = batch[self.hparams.x_label].float()
            condition = batch[self.c_label].float()
        else:
            x = batch[self.hparams.x_label].float()
            x_channels = [channel for channel in range(x.shape[1])
                          if channel not in self.c_label]

            x = x[:, x_channels]
            condition = x[:, self.c_label]

        return x, {"condition": condition}
