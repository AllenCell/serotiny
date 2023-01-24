from typing import Optional, Sequence
from omegaconf import DictConfig

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss as Loss
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from serotiny.models.base_model import BaseModel
from .priors import Prior, IsotropicGaussianPrior


def _latent_compose_function(z_parts, **kwargs):
    return z_parts


class BaseVAE(BaseModel):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        latent_dim: int,
        x_label: str,
        beta: float = 1.0,
        id_label: Optional[str] = None,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        lr_scheduler: Optional[LRScheduler] = None,
        reconstruction_loss: Loss = nn.MSELoss(reduction="none"),
        prior: Optional[Sequence[Prior]] = None,
        latent_compose_function=None,
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
        x_label: Optional[str] = None
        id_label: Optional[str] = None
        optimizer: torch.optim.Optimizer
            Optimizer to use
        beta: float = 1.0
            Beta parameter - the weight of the KLD term in the loss function
        reconstruction_loss: Loss
            Loss to be used for reconstruction. Can be a PyTorch loss or a class
            that respects the same interface,
            i.e. subclasses torch.nn.modules._Loss
        prior: Optional[Sequence[AbstractPrior]]
            List of prior specifications to use for latent space
        """
        super().__init__()

        self.reconstruction_loss = reconstruction_loss

        if not isinstance(encoder, (dict, DictConfig)):
            assert x_label is not None
            encoder = {x_label: encoder}
        self.encoder = nn.ModuleDict(encoder)

        if not isinstance(decoder, (dict, DictConfig)):
            assert x_label is not None
            decoder = {x_label: decoder}
        self.decoder = nn.ModuleDict(decoder)

        self.beta = beta
        self.latent_dim = latent_dim

        if prior is None:
            prior = IsotropicGaussianPrior()

        if not isinstance(prior, (dict, DictConfig)):
            assert x_label is not None
            prior = {x_label: prior}

        self.prior = nn.ModuleDict(prior)

        if latent_compose_function is None:
            latent_compose_function = _latent_compose_function
        self.latent_compose_function = latent_compose_function

    def calculate_rcl(self, x_hat, x, key):
        rcl_per_input_dimension = self.reconstruction_loss[key](x_hat[key], x[key])
        return rcl_per_input_dimension

    def calculate_elbo(self, x, x_hat, z):
        rcl_per_input_dimension = {}
        rcl_reduced = {}
        for key in x_hat.keys():
            rcl_per_input_dimension[key] = self.calculate_rcl(x_hat, x, key)
            if len(rcl_per_input_dimension[key].shape) > 0:
                rcl = (
                    rcl_per_input_dimension[key]
                    # flatten
                    .view(rcl_per_input_dimension[key].shape[0], -1)
                    # and sum across each batch element's dimensions
                    .sum(dim=1)
                )

                rcl_reduced[key] = rcl.mean()
            else:
                rcl_reduced[key] = rcl_per_input_dimension[key]

        kld_per_part = {
            part: self.prior[part](z_part, mode="kl", reduction="none")
            for part, z_part in z.items()
        }

        kld_per_part_summed = {
            part: kl.sum(dim=-1).mean() for part, kl in kld_per_part.items()
        }

        total_kld = sum(kld_per_part_summed.values())
        return (
            sum(rcl_reduced.values()) + self.beta * total_kld,
            rcl_reduced,
            total_kld,
            kld_per_part,
        )

    def sample_z(self, z_parts_params):
        return {
            part: prior(z_parts_params[part], mode="sample")
            for part, prior in self.prior.items()
        }

    def encode(self, batch):
        return {
            part: encoder(batch[part].float()) for part, encoder in self.encoder.items()
        }

    def decode(self, z_parts):
        z = self.latent_compose_function(z_parts)

        return (
            {part: decoder(z[part].float()) for part, decoder in self.decoder.items()},
            z,
        )

    def forward(self, batch, decode=False, compute_loss=False, **kwargs):

        z_parts_params = self.encode(batch)

        z_parts = self.sample_z(z_parts_params)

        x_hat, z_composed = self.decode(z_parts)

        if not decode:
            return z_parts_params, z_composed

        if not compute_loss:
            return x_hat, z_parts, z_parts_params, z_composed

        (
            loss,
            reconstruction_loss,
            kld_loss,
            kld_per_part,
        ) = self.calculate_elbo(batch, x_hat, z_parts_params)

        return (
            x_hat,
            z_parts,
            z_parts_params,
            z_composed,
            loss,
            reconstruction_loss,
            kld_loss,
            kld_per_part,
        )

    def log_metrics(self, stage, results, logger, batch_size):
        on_step = (stage == "val") | (stage == "train")

        for key, value in results.items():
            if (len(value.shape) == 0) | (
                len(value.shape) == 1 and value.shape[0] == 1  # noqa
            ):
                self.log(
                    f"{stage} {key}",
                    value.squeeze(),
                    logger=logger,
                    on_step=on_step,
                    on_epoch=True,
                    batch_size=batch_size,
                    sync_dist=True,
                )

    def make_results_dict(
        self,
        stage,
        batch,
        loss,
        reconstruction_loss,
        kld_loss,
        kld_per_part,
        z_parts,
        z_parts_params,
        z_composed,
    ):

        results = {
            "loss": loss,
            f"{stage}_loss": loss.detach(),  # for epoch end logging purposes
            "kld_loss": kld_loss.detach(),
        }

        for part, z_comp_part in z_composed.items():
            results.update(
                {
                    f"z_composed/{part}": z_comp_part.detach(),
                }
            )

        for part, recon_part in reconstruction_loss.items():
            results.update(
                {
                    f"reconstruction_loss/{part}": recon_part.detach(),
                }
            )

        for part, z_part in z_parts.items():
            results.update(
                {
                    f"z_parts/{part}": z_part.detach(),
                    f"z_parts_params/{part}": z_parts_params[part].detach(),
                    f"kld/{part}": kld_per_part[part].detach(),
                }
            )

        if self.hparams.id_label is not None:
            if self.hparams.id_label in batch:
                ids = batch[self.hparams.id_label].detach()
                results.update({self.hparams.id_label: ids, "id": ids})

        return results

    def _step(self, stage, batch, batch_idx, logger):
        (
            x_hat,
            z_parts,
            z_parts_params,
            z_composed,
            loss,
            reconstruction_loss,
            kld_loss,
            kld_per_part,
        ) = self.forward(batch, decode=True, compute_loss=True)

        results = self.make_results_dict(
            stage,
            batch,
            loss,
            reconstruction_loss,
            kld_loss,
            kld_per_part,
            z_parts,
            z_parts_params,
            z_composed,
        )

        self.log_metrics(stage, results, logger, batch[self.hparams.x_label].shape[0])

        return results
