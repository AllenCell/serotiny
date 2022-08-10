import inspect
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss as Loss

from serotiny.models.base_model import BaseModel
from .priors import Prior, IsotropicGaussianPrior

Array = Union[torch.Tensor, np.ndarray, Sequence[float]]


def get_args(encoder):
    args = []
    if isinstance(encoder, nn.Sequential):
        for i in range(len(encoder)):
            args.append(inspect.getfullargspec(encoder[i].forward).args)
        args = [item for sublist in args for item in sublist]
    else:
        args = inspect.getfullargspec(encoder.forward).args
    return args


class BaseVAE(BaseModel):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        latent_dim: Union[int, Sequence[int]],
        beta: float,
        x_label: str,
        id_label: Optional[str] = None,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        loss_mask_label: Optional[str] = None,
        reconstruction_loss: Loss = nn.MSELoss(reduction="none"),
        reconstruction_reduce: str = "sum",
        priors: Optional[Sequence[Prior]] = None,
        cache_outputs: Sequence = ("test",),
        **kwargs,
    ):
        """Instantiate a basic VAE model.

        Parameters
        ----------
        encoder: nn.Module
            Encoder network
        decoder: nn.Module
            Decoder network
        optimizer: torch.optim.Optimizer
            Optimizer to use
        beta: float
            Beta parameter - the weight of the KLD term in the loss function
        x_label: str
            String label used to retrieve X from the batches
        reconstruction_loss: Loss
            Loss to be used for reconstruction. Can be a PyTorch loss or a class
            that respects the same interface,
            i.e. subclasses torch.nn.modules._Loss
        priors: Optional[Sequence[AbstractPrior]]
            List of prior specifications to use for latent space
        """
        super().__init__()

        self.reconstruction_reduce = reconstruction_reduce
        self.reconstruction_loss = reconstruction_loss
        self.encoder = encoder
        self.decoder = decoder

        self.beta = beta
        self.latent_dim = latent_dim

        self.encoder_args = inspect.getfullargspec(self.encoder.forward).args
        self.decoder_args = inspect.getfullargspec(self.decoder.forward).args

        if priors is None:
            priors = [IsotropicGaussianPrior()]

        self.priors = nn.ModuleList(priors)

    def calculate_elbo(self, x, x_hat, z, mask=None):
        rcl_per_input_dimension = self.reconstruction_loss(x_hat, x)

        if mask is not None:
            rcl_per_input_dimension = rcl_per_input_dimension * mask
            normalizer = mask.view(mask.shape[0], -1).sum(dim=1)
        else:
            normalizer = np.prod(x.shape[1:])

        rcl = (
            rcl_per_input_dimension
            # flatten
            .view(rcl_per_input_dimension.shape[0], -1)
            # and sum across each batch element's dimensions
            .sum(dim=1)
        )

        if self.reconstruction_reduce == "mean":
            rcl = rcl / normalizer

        rcl = rcl.mean()

        kld = torch.sum([prior(z, mode="kl") for prior in self.priors])

        return (
            rcl + self.beta * kld,
            rcl,
            kld,
        )

    def parse_batch(self, batch):
        if self.hparams.loss_mask_label is not None:
            mask = batch[self.hparams.loss_mask_label].float()
        else:
            mask = None
        return batch[self.hparams.x_label].float(), dict(mask=mask)

    def sample_z(self, z):
        return torch.cat([prior(z, mode="sample") for prior in self.priors], dim=1)

    def encode(self, x, **kwargs):
        return self.encoder(
            x, **{k: v for k, v in kwargs.items() if k in self.encoder_args}
        )

    def decode(self, z, **kwargs):
        return self.decoder(
            z, **{k: v for k, v in kwargs.items() if k in self.decoder_args}
        )

    def forward(self, x, decode=False, compute_loss=False, **kwargs):
        z_params = self.encode(x, **kwargs)
        if not decode:
            return z_params

        z = self.sample_z(z_params)
        x_hat = self.decode(z, **kwargs)

        if not compute_loss:
            return x_hat, z, z_params

        (
            loss,
            reconstruction_loss,
            kld_loss,
            rcl_per_input_dimension,
            kld_per_latent_dimension,
        ) = self.calculate_elbo(x, x_hat, z_params, mask=kwargs.get("mask", None))

        return (
            x_hat,
            z,
            z_params,
            loss,
            reconstruction_loss,
            kld_loss,
        )

    def log_metrics(self, stage, reconstruction_loss, kld_loss, loss, logger):
        on_step = stage == "train"

        self.log(
            f"{stage} reconstruction loss",
            reconstruction_loss,
            logger=logger,
            on_step=on_step,
            on_epoch=True,
        )
        self.log(
            f"{stage} kld loss", kld_loss, logger=logger, on_step=on_step, on_epoch=True
        )
        self.log(f"{stage}_loss", loss, logger=logger, on_step=on_step, on_epoch=True)

    def _step(self, stage, batch, batch_idx, logger):

        x = self.parse_batch(batch)

        if isinstance(x, tuple):
            x, forward_kwargs = x
        else:
            forward_kwargs = dict()

        (
            x_hat,
            z,
            z_params,
            loss,
            reconstruction_loss,
            kld_loss,
            rcl_per_input_dimension,
            kld_per_latent_dimension,
        ) = self.forward(x, decode=True, compute_loss=True, **forward_kwargs)

        self.log_metrics(stage, reconstruction_loss, kld_loss, loss, logger)

        results = {
            "loss": loss,
            f"{stage}_loss": loss.detach().cpu(),  # for epoch end logging purposes
            "reconstruction_loss": reconstruction_loss.detach().cpu(),
            "kld_loss": kld_loss.detach().cpu(),
            "batch_idx": batch_idx,
            "z": z.detach().cpu(),
            "z_params": z_params.detach().cpu(),
            "kld_per_latent_dimension": kld_per_latent_dimension.detach().float().cpu(),
            "rcl_per_input_dimension": rcl_per_input_dimension.detach().float().cpu(),
        }

        if stage == "test":
            results.update(
                {
                    "x_hat": x_hat.detach().cpu(),
                    "x": x.detach().cpu(),
                }
            )

        if self.hparams.id_label is not None:
            if self.hparams.id_label in batch:
                ids = batch[self.hparams.id_label].detach().cpu()
                results.update({self.hparams.id_label: ids, "id": ids})

        return results
