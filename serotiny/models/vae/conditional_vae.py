"""
Base conditional variational autoencoder module, implemented as a
Pytorch Lightning module
"""

from typing import Sequence, Callable, Union, Optional
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

from serotiny.losses.elbo import calculate_elbo
from serotiny.models._utils import find_optimizer, find_lr_scheduler
from serotiny.models.vae.basic_vae import BasicVAE

Array = Union[torch.Tensor, np.array]

class ConditionalVAE(BasicVAE):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        optimizer: opt.Optimizer,
        reconstruction_loss: Callable,
        lr: float,
        beta: float,
        x_label: str,
        c_label: Union[str, int, Sequence[int]],
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
            Encoder network. (Untested with other than CBVAEEncoder)
        decoder: nn.Module
            Decoder network. (Untested with other than CBVAEDecoder)
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

        super().__init__(encoder, decoder, optimizer, reconstruction_loss,
                         lr, beta, x_label, prior_mode, prior_logvar,
                         learn_prior_logvar)

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
            if condition_mode == "label":
                x = batch[self.hparams.x_label].float()
                condition = batch[self.hparams.c_label].float()
            else:
                x = batch[self.hparams.x_label].float()
                x_channels = [channel for channel in range(x.shape[1])
                              if channel not in self.c_label]

                x = x[:, x_channels]
                condition = x[:, self.c_label]

            return x, {"condition": condition}
