from typing import Callable, Union, Optional, Sequence
import inspect

import logging
logger = logging.getLogger("lightning")
logger.propagate = False

import numpy as np
import torch
import torch.optim as opt
import torch.nn as nn
from torch.nn.modules.loss import _Loss as Loss

import pytorch_lightning as pl
from pytorch_lightning.utilities.parsing import get_init_args

from serotiny.losses.elbo import calculate_elbo
from serotiny.models._utils import find_optimizer, find_lr_scheduler
from serotiny.networks.mlp import MLP

Array = Union[torch.Tensor, np.array, Sequence[float]]

class BaseVAE(pl.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        optimizer: str,
        lr: float,
        beta: float,
        x_label: str,
        recon_loss: Loss = torch.nn.MSELoss,
        prior_mode: str = "isotropic",
        prior_logvar: Optional[Array] = None,
        learn_prior_logvar: bool = False,
    ):
        """
        Instantiate a basic VAE model

        Parameters
        ----------
        encoder: nn.Module
            Encoder network.
        decoder: nn.Module
            Decoder network.
        optimizer: opt.Optimizer
            Optimizer to use
        lr: float
            Learning rate for training
        beta: float
            Beta parameter - the weight of the KLD term in the loss function
        x_label: str
            String label used to retrieve X from the batches
        recon_loss: Loss
            Loss to be used for reconstruction. Can be a PyTorch loss or a class
            that respects the same interface, i.e. subclasses torch.nn.modules._Loss
        prior_mode: str
            String to determine which type of prior to use. (Only "isotropic" and
            "anisotropic" currently supported)
        prior_logvar: Optional[Array]
            Array of values to be used as either the fixed value or initialization
            value for the diagonal of the prior covariance matrix
        learn_prior_logvar: bool
            Boolean flag to determine whether to learn the prior log-variances
        """
        super().__init__()

        # store init args except for encoder & decoder, to avoid copying
        # large objects unnecessarily
        frame = inspect.currentframe()
        init_args = get_init_args(frame)
        self.save_hyperparameters(
            *[arg for arg in init_args if arg not in ["encoder", "decoder"]]
        )

        self.recon_loss = recon_loss

        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta

        self.prior_mode = prior_mode
        self.prior_mean = None
        if prior_logvar is not None:
            prior_logvar = torch.Tensor(prior_logvar)
        self.prior_logvar = prior_logvar

        if prior_mode not in ["isotropic", "anisotropic"]:
            raise NotImplementedError(f"KLD mode '{prior_mode}' not implemented")

        if prior_mode == "anisotropic":
            self.prior_mean = torch.zeros(self.embedding_dim)
            if prior_logvar is None:
                self.prior_logvar = torch.zeros(self.embedding_dim)
            else:
                self.prior_logvar = torch.tensor(prior_logvar)
            # if learn_prior_logvar:
            self.prior_logvar = nn.Parameter(
                self.prior_logvar, requires_grad=learn_prior_logvar
            )

        self.encoder_args = inspect.getargspec(self.encoder.forward).args
        self.decoder_args = inspect.getargspec(self.decoder.forward).args

    def parse_batch(self, batch):
        return batch[self.hparams.x_label].float()

    def sample_z(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu)

    def forward(self, x, **kwargs):
        mu_logvar = self.encoder(x, **{k:v for k,v in kwargs.items()
                                       if k in self.encoder_args})

        mu, logvar = torch.split(mu_logvar, mu_logvar.shape[1] // 2, dim=1)
        assert mu.shape == logvar.shape

        z = self.sample_z(mu, logvar)

        x_hat = self.decoder(z, **{k:v for k,v in kwargs.items()
                                   if k in self.decoder_args})

        (loss, recon_loss, kld_loss,
         rcl_per_input_dimension, kld_per_latent_dimension) = calculate_elbo(
            x,
            x_hat,
            mu,
            logvar,
            self.beta,
            recon_loss=self.recon_loss,
            mode=self.prior_mode,
            prior_mu=(None if self.prior_mean is None else self.prior_mean.type_as(mu)),
            prior_logvar=self.prior_logvar,
        )

        batch_size = x.shape[0]

        return (
            x_hat,
            mu,
            loss / batch_size,
            recon_loss / batch_size,
            kld_loss / batch_size,
            rcl_per_input_dimension,
            kld_per_latent_dimension
        )

    def _step(self, stage, batch, batch_idx, logger):
        x = self.parse_batch(batch)
        if isinstance(x, tuple):
            x, forward_kwargs = x
        else:
            forward_kwargs = dict()

        (_, _, loss, recon_loss, kld_loss,
         rcl_per_input_dimension,
         kld_per_latent_dimension) = self.forward(x, **forward_kwargs)

        self.log(f"{stage} reconstruction loss", recon_loss, logger=logger)
        self.log(f"{stage} kld loss", kld_loss, logger=logger)
        self.log(f"{stage}_loss", loss, logger=logger)

        return {
            "loss": loss,
            f"{stage}_loss": loss,  # for epoch end logging purposes
            "recon_loss": recon_loss,
            "kld_loss": kld_loss,
            #"kld_per_latent_dimension": kld_per_latent_dimension,
            #"rcl_per_input_dimension": rcl_per_input_dimension,
            "batch_idx": batch_idx,
        }

    def training_step(self, batch, batch_idx):
        return self._step("train", batch, batch_idx, logger=False)

    def validation_step(self, batch, batch_idx):
        return self._step("val", batch, batch_idx, logger=True)

    def test_step(self, batch, batch_idx):
        return self._step("test", batch, batch_idx, logger=True)

    def configure_optimizers(self):
        optimizer_class = find_optimizer(self.hparams.optimizer)
        optimizer = optimizer_class(self.parameters(), lr=self.hparams.lr)

        return {
            "optimizer": optimizer,
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
            "strict": True,
        }

