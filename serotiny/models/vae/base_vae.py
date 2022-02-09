from typing import Union, Optional, Sequence, Dict

import logging
import inspect

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss as Loss

import pytorch_lightning as pl
from pytorch_lightning.utilities.parsing import get_init_args

from serotiny.models.utils import find_optimizer

from serotiny.losses.kl_divergence import diagonal_gaussian_kl, isotropic_gaussian_kl

Array = Union[torch.Tensor, np.array, Sequence[float]]
logger = logging.getLogger("lightning")
logger.propagate = False


class BaseVAE(pl.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        latent_dim: Union[int, Sequence[int]],
        beta: float,
        x_label: str,
        optimizer = torch.optim.Adam,
        loss_mask_label: Optional[str] = None,
        reconstruction_loss: Loss = nn.MSELoss(reduction="none"),
        reconstruction_reduce: str = "sum",
        prior_mode: str = "isotropic",
        prior_logvar: Optional[Array] = None,
        learn_prior_logvar: bool = False,
    ):
        """
        Instantiate a basic VAE model

        Parameters
        ----------
        encoder: Union[nn.Module, dict]
            Encoder network. If `dict`, expects a class dict, to be used for
            instantiating the given class.
        decoder: Union[nn.Module, str]
            Decoder network. If `dict`, expects a class dict, to be used for
            instantiating the given class.
        optimizer: str
            Optimizer to use
        beta: float
            Beta parameter - the weight of the KLD term in the loss function
        x_label: str
            String label used to retrieve X from the batches
        reconstruction_loss: Loss
            Loss to be used for reconstruction. Can be a PyTorch loss or a class
            that respects the same interface,
            i.e. subclasses torch.nn.modules._Loss
        prior_mode: str
            String to determine which type of prior to use.
            (Only "isotropic" and
            "anisotropic" currently supported)
        prior_logvar: Optional[Array]
            Array of values to be used as either the fixed value or initialization
            value for the diagonal of the prior covariance matrix
        learn_prior_logvar: bool
            Boolean flag to determine whether to learn the prior log-variances
        """
        super().__init__()

        frame = inspect.currentframe()
        init_args = get_init_args(frame)
        self.save_hyperparameters(
            *[arg for arg in init_args
              if arg not in ["encoder", "decoder", "optimizer"]]
        )

        self.reconstruction_reduce = reconstruction_reduce
        self.reconstruction_loss = reconstruction_loss
        self.encoder = encoder
        self.decoder = decoder

        self.beta = beta
        self.latent_dim = latent_dim

        self.prior_mode = prior_mode
        self.prior_mu = None
        if prior_logvar is not None:
            prior_logvar = torch.Tensor(prior_logvar)
        self.prior_logvar = prior_logvar

        self.optimizer = optimizer

        if prior_mode not in ["isotropic", "anisotropic"]:
            raise NotImplementedError(f"KLD mode '{prior_mode}' not implemented")

        if prior_mode == "anisotropic":
            # with an anisotropic gaussian prior, we allow for the prior log-variance
            # to be different than all-ones, and additionally to be learned, if desired
            self.prior_mu = torch.zeros(self.latent_dim)
            if prior_logvar is None:
                self.prior_logvar = torch.zeros(self.latent_dim)
            else:
                self.prior_logvar = torch.tensor(prior_logvar)

            if learn_prior_logvar:
                self.prior_logvar = nn.Parameter(self.prior_logvar, requires_grad=True)
            else:
                self.prior_logvar.requires_grad = False

        self.encoder_args = inspect.getfullargspec(self.encoder.forward).args
        self.decoder_args = inspect.getfullargspec(self.decoder.forward).args

        self._cached_outputs = dict()

    def calculate_elbo(self, x, x_hat, mu, logvar, mask=None):
        rcl_per_input_dimension = self.reconstruction_loss(x_hat, x)
        if mask is not None:
            rcl_per_input_dimension = rcl_per_input_dimension * mask
            normalizer = mask.view(mask.shape[0], -1).sum(dim=1)
        else:
            normalizer = sum(x.shape[1:])

        rcl = (
            rcl_per_input_dimension
            # flatten
            .view(rcl_per_input_dimension.shape[0], -1)
            # and sum per batch element.
            .sum(dim=1)
        )

        if self.reconstruction_reduce == "mean":
            rcl = rcl / normalizer

        rcl = rcl.mean()

        if self.prior_mode == "isotropic":
            kld_per_dimension = isotropic_gaussian_kl(mu, logvar)
        elif self.prior_mode == "anisotropic":
            prior_mu = ((None if self.prior_mu is None else self.prior_mu.type_as(mu)),)
            kld_per_dimension = diagonal_gaussian_kl(
                mu, prior_mu, logvar, self.prior_logvar
            )
        else:
            raise NotImplementedError(f"KLD mode '{self.prior_mode}' not implemented")

        kld = kld_per_dimension.sum(dim=1).mean()

        return (
            rcl + self.beta * kld,
            rcl,
            kld,
            rcl_per_input_dimension,
            kld_per_dimension,
        )

    def parse_batch(self, batch):
        if self.hparams.loss_mask_label is not None:
            mask = batch[self.hparams.loss_mask_label].float()
        else:
            mask = None
        return batch[self.hparams.x_label].float(), dict(mask=mask)

    def sample_z(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu)

    def forward(self, x, **kwargs):
        mu_logvar = self.encoder(
            x, **{k: v for k, v in kwargs.items() if k in self.encoder_args}
        )

        mu, logvar = torch.split(mu_logvar, mu_logvar.shape[1] // 2, dim=1)

        # logvar is clamped here to avoid gradient explosion, since it
        # will get exponentiated when calculating kl_divergences
        #logvar = logvar.clamp(min=0, max=50)
        assert mu.shape == logvar.shape

        z = self.sample_z(mu, logvar)

        x_hat = self.decoder(
            z, **{k: v for k, v in kwargs.items() if k in self.decoder_args}
        )

        (
            loss,
            reconstruction_loss,
            kld_loss,
            rcl_per_input_dimension,
            kld_per_latent_dimension,
        ) = self.calculate_elbo(x, x_hat, mu, logvar)

        return (
            x_hat,
            mu,
            logvar,
            loss,
            reconstruction_loss,
            kld_loss,
            rcl_per_input_dimension,
            kld_per_latent_dimension,
        )

    def _step(self, stage, batch, batch_idx, logger):
        x = self.parse_batch(batch)

        if isinstance(x, tuple):
            x, forward_kwargs = x
        else:
            forward_kwargs = dict()

        (
            x_hat,
            mu,
            _,
            loss,
            reconstruction_loss,
            kld_loss,
            rcl_per_input_dimension,
            kld_per_latent_dimension,
        ) = self.forward(x, **forward_kwargs)

        self.log(f"{stage} reconstruction loss", reconstruction_loss, logger=logger)
        self.log(f"{stage} kld loss", kld_loss, logger=logger)
        self.log(f"{stage}_loss", loss, logger=logger)

        results = {
            "loss": loss,
            f"{stage}_loss": loss.detach().cpu(),  # for epoch end logging purposes
            "reconstruction_loss": reconstruction_loss.detach().cpu(),
            "kld_loss": kld_loss.detach().cpu(),
            "batch_idx": batch_idx,
            "x_hat": x_hat.detach().cpu(),
            "x": x.detach().cpu(),
            "mu": mu.detach().cpu(),
            "kld_per_latent_dimension": kld_per_latent_dimension.detach().float().cpu(),
            "rcl_per_input_dimension": rcl_per_input_dimension.detach().float().cpu(),
        }

        return results

    def training_step(self, batch, batch_idx):
        return self._step("train", batch, batch_idx, logger=True)

    def validation_step(self, batch, batch_idx):
        return self._step("val", batch, batch_idx, logger=True)

    def test_step(self, batch, batch_idx):
        return self._step("test", batch, batch_idx, logger=False)

    def _epoch_end(self, split, outputs):
        self._cached_outputs[split] = outputs

    def train_epoch_end(self, outputs):
        self._epoch_end("train", outputs)


    def validation_epoch_end(self, outputs):
        self._epoch_end("val", outputs)


    def test_epoch_end(self, outputs):
        self._epoch_end("test", outputs)


    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())
        return {
            "optimizer": optimizer,
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
            "strict": True,
        }
