from typing import Optional, Sequence
from omegaconf import DictConfig

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss as Loss
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from serotiny.models.base_model import BaseModel
from .priors import Prior, IsotropicGaussianPrior


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
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler = torch.optim.lr_scheduler.StepLR,
        loss_mask_label: Optional[str] = None,
        reconstruction_loss: torch.nn.modules.loss._Loss = nn.MSELoss(reduction="none"),
        prior: Optional[Sequence[Prior]] = None,
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

        self.decoder = decoder

        self.beta = beta
        self.latent_dim = latent_dim

        if prior is None:
            prior = IsotropicGaussianPrior()

        if not isinstance(prior, (dict, DictConfig)):
            assert x_label is not None
            prior = {x_label: prior}

        self.prior = nn.ModuleDict(prior)

    def calculate_rcl(self, x_hat, x):
        rcl_per_input_dimension = self.reconstruction_loss(x_hat, x)
        return rcl_per_input_dimension

    def calculate_elbo(self, x, x_hat, z, mask=None):
        rcl_per_input_dimension = self.calculate_rcl(x_hat, x)

        if mask is not None:
            rcl_per_input_dimension = rcl_per_input_dimension * mask

        rcl = (
            rcl_per_input_dimension
            # flatten
            .view(rcl_per_input_dimension.shape[0], -1)
            # and sum across each batch element's dimensions
            .sum(dim=1)
        )

        rcl = rcl.mean()
        kld_per_part = {
            part: self.prior[part](z_part, mode="kl") for part, z_part in z.items()
        }

        kld_per_part = {
            part: torch.sum(this_kld_part.view(-1,1), dim=1).float().mean() for part, this_kld_part in kld_per_part.items() 
        }

        return (
            rcl + self.beta * sum(kld_per_part.values()),
            rcl,
            sum(kld_per_part.values()),
            kld_per_part,
        )

    def sample_z(self, z_parts_params):
        return {
            part: prior(z_parts_params[part], mode="sample")
            for part, prior in self.prior.items()
        }

    def encode(self, batch):
        return {part: encoder(batch[part]) for part, encoder in self.encoder.items()}

    def latent_compose_function(self, z_parts, **kwargs):
        return torch.cat(z_parts.values(), dim=1)

    def decode(self, z_parts):
        z = self.latent_compose_function(z_parts)
        return self.decoder(z), z
    
    def parse_batch(self, batch):
        return batch

    def forward(self, batch, decode=False, compute_loss=False):

        batch = self.parse_batch(batch)

        z_parts_params = self.encode(batch)

        if not decode:
            return z_parts_params

        z_parts = self.sample_z(z_parts_params)

        x_hat, z_composed = self.decode(z_parts)

        if not compute_loss:
            return x_hat, z_parts, z_parts_params

        mask = batch.get(self.hparams.get("loss_mask_label"))

        (loss, reconstruction_loss, kld_loss, kld_per_part,) = self.calculate_elbo(
            batch[self.hparams.x_label], x_hat, z_parts_params, mask=mask
        )

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
        x_hat,
    ):
        results = {
            "loss": loss,
            f"{stage}_loss": loss.detach().cpu(),  # for epoch end logging purposes
            "reconstruction_loss": reconstruction_loss.detach().cpu(),
            "kld_loss": kld_loss.detach().cpu(),
            "z_composed": z_composed.detach().cpu(),
        }
        # import ipdb
        # ipdb.set_trace()
        for part, z_part in z_parts.items():
            results.update(
                {
                    f"z_parts/{part}": z_part.detach().cpu(),
                    f"z_parts_params/{part}": z_parts_params[part].detach().cpu(),
                    f"kld/{part}": kld_per_part[part].detach().float().cpu(),
                }
            )

        if stage == "test":
            results.update(
                {
                    "x_hat": x_hat.detach().cpu(),
                }
            )
            for k, v in batch.items():
                results[k] = v.detach.cpu()

        if self.hparams.id_label is not None:
            if self.hparams.id_label in batch:
                ids = batch[self.hparams.id_label].detach().cpu()
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

        self.log_metrics(stage, reconstruction_loss, kld_loss, loss, logger)

        return self.make_results_dict(
            stage,
            batch,
            loss,
            reconstruction_loss,
            kld_loss,
            kld_per_part,
            z_parts,
            z_parts_params,
            z_composed,
            x_hat,
        )
