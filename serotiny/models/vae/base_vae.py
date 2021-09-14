from typing import Union, Optional, Sequence, Dict
import inspect

import logging

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss as Loss

import pytorch_lightning as pl
from pytorch_lightning.utilities.parsing import get_init_args

from serotiny.losses.elbo import calculate_elbo
from serotiny.models._utils import find_optimizer
from serotiny.utils import load_config
from serotiny.utils.dynamic_imports import _bind


Array = Union[torch.Tensor, np.array, Sequence[float]]
logger = logging.getLogger("lightning")
logger.propagate = False


class BaseVAE(pl.LightningModule):
    def __init__(
        self,
        encoder: Union[nn.Module, Dict],
        decoder: Union[nn.Module, Dict],
        latent_dim: Union[int, Sequence[int]],
        optimizer: str,
        beta: float,
        x_label: str,
        lr: float = 1e-3,
        loss_mask_label: Optional[str] = None,
        recon_loss: Union[Loss, Dict] = nn.MSELoss(reduction="none"),
        recon_reduce: str = "mean",
        prior_mode: str = "isotropic",
        prior_logvar: Optional[Array] = None,
        learn_prior_logvar: bool = False,
    ):
        """
                Instantiate a basic VAE model

                Parameters
                ----------
                encoder: Union[nn.Module, dict]
        `           Encoder network. If `dict`, expects a class dict, to be used for
                    instantiating the given class.
                decoder: Union[nn.Module, str]
                    Decoder network. If `dict`, expects a class dict, to be used for
                    instantiating the given class.
                optimizer: str
                    Optimizer to use
                lr: float
                    Learning rate for training
                beta: float
                    Beta parameter - the weight of the KLD term in the loss function
                x_label: str
                    String label used to retrieve X from the batches
                recon_loss: Loss
                    Loss to be used for reconstruction. Can be a PyTorch loss or a class
                    that respects the same interface,
                    i.e. subclasses torch.nn.modules._Loss
                prior_mode: str
                    String to determine which type of prior to use.
                    (Only "isotropic" and
                    "anisotropic" currently supported)
                prior_logvar: Optional[Array]
                    Array of values to be used as either the fixed value or
                    initialization
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

        if isinstance(recon_loss, dict):
            recon_loss = load_config(recon_loss)
        if isinstance(encoder, dict):
            encoder = load_config(encoder)
        if isinstance(decoder, str):
            decoder = load_config(decoder)

        self.recon_reduce = recon_reduce

        self.recon_loss = recon_loss
        self.encoder = encoder
        self.decoder = decoder

        self.beta = beta
        self.latent_dim = latent_dim

        self.prior_mode = prior_mode
        self.prior_mean = None
        if prior_logvar is not None:
            prior_logvar = torch.Tensor(prior_logvar)
        self.prior_logvar = prior_logvar

        if prior_mode not in ["isotropic", "anisotropic"]:
            raise NotImplementedError(f"KLD mode '{prior_mode}' not implemented")

        if prior_mode == "anisotropic":
            self.prior_mean = torch.zeros(self.latent_dim)
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

    def parse_batch(self, batch):
        if self.hparams.loss_mask_label is not None:
            mask = batch[self.hparams.loss_mask_label].float()
        else:
            mask = None
        return batch[self.hparams.x_label].float(), dict(mask=mask)


    def sample_z(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu)

    def forward(self, x, **kwargs):
        mu_logvar = self.encoder(
            x, **{k: v for k, v in kwargs.items() if k in self.encoder_args}
        )

        mu, logvar = torch.split(mu_logvar, mu_logvar.shape[1] // 2, dim=1)
        logvar = logvar.clamp(min=0, max=50)
        assert mu.shape == logvar.shape

        z = self.sample_z(mu, logvar)

        x_hat = self.decoder(
            z, **{k: v for k, v in kwargs.items() if k in self.decoder_args}
        )


        (
            loss,
            recon_loss,
            kld_loss,
            rcl_per_input_dimension,
            kld_per_latent_dimension,
        ) = calculate_elbo(
            x,
            x_hat,
            mu,
            logvar,
            self.beta,
            recon_loss=self.recon_loss,
            recon_reduce=self.recon_reduce,
            mode=self.prior_mode,
            prior_mu=(None if self.prior_mean is None else self.prior_mean.type_as(mu)),
            prior_logvar=self.prior_logvar,
        )

        return (
            x_hat,
            mu,
            logvar,
            loss,
            recon_loss,
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
            _,
            mu,
            _,
            loss,
            recon_loss,
            kld_loss,
            rcl_per_input_dimension,
            kld_per_latent_dimension,
        ) = self.forward(x, **forward_kwargs)

        self.log(f"{stage} reconstruction loss", recon_loss, logger=logger)
        self.log(f"{stage} kld loss", kld_loss, logger=logger)
        self.log(f"{stage}_loss", loss, logger=logger)

        results = {
            "loss": loss,
            f"{stage}_loss": loss.detach(),  # for epoch end logging purposes
            "recon_loss": recon_loss.detach(),
            "kld_loss": kld_loss.detach(),
            "batch_idx": batch_idx,
        }

        if stage in ("test", "val"):
            results.update({
                "mu": mu.detach(),
                "kld_per_latent_dimension": kld_per_latent_dimension.detach().float(),
                "rcl_per_input_dimension": rcl_per_input_dimension.detach().float(),
            })

        return results

    def training_step(self, batch, batch_idx):
        return self._step("train", batch, batch_idx, logger=True)

    def validation_step(self, batch, batch_idx):
        return self._step("val", batch, batch_idx, logger=True)

    def test_step(self, batch, batch_idx):
        return self._step("test", batch, batch_idx, logger=False)

    def configure_optimizers(self):
        if isinstance(self.hparams.optimizer, str):
            optimizer_class = find_optimizer(self.hparams.optimizer)
            optimizer = optimizer_class(self.parameters(), lr=self.hparams.lr)
        elif isinstance(self.hparams.optimizer, dict):
            optimizer = load_config(self.hparams.optimizer)
            optimizer = optimizer(self.parameters())
        elif isinstance(self.hparams.optimizer, _bind):
            optimizer = self.hparams.optimizer(self.parameters())

        else:
            raise TypeError

        return {
            "optimizer": optimizer,
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
            "strict": True,
        }
