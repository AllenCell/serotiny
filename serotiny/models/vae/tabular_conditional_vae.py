"""
Tabular conditional variational autoencoder module, implemented as a
Pytorch Lightning module
"""

from typing import Sequence, Union, Optional

import logging

logger = logging.getLogger("lightning")
logger.propagate = False

import numpy as np
import torch
from torch.nn.modules.loss import _Loss as Loss

from serotiny.networks.mlp import MLP
from .conditional_vae import ConditionalVAE

Array = Union[torch.Tensor, np.array]


class TabularConditionalVAE(ConditionalVAE):
    def __init__(
        self,
        x_dim: int,
        latent_dim: int,
        c_dim: int,
        hidden_layers: Sequence[int],
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
        reconstruction_loss: Callable
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

        encoder = MLP(
            x_dim + c_dim,
            2 * latent_dim,
            hidden_layers=hidden_layers,
        )

        decoder = MLP(
            latent_dim + c_dim,
            x_dim,
            hidden_layers=hidden_layers,
        )

        super().__init__(
            encoder,
            decoder,
            latent_dim,
            optimizer,
            lr,
            beta,
            x_label,
            c_label,
            recon_loss,
            condition_mode,
            prior_mode,
            prior_logvar,
            learn_prior_logvar,
        )

    def parse_batch(self, batch):
        batch = super().parse_batch(batch)
        batch[1]["x2"] = batch[1].pop("condition")
        return batch
