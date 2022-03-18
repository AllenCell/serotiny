from typing import Union, Optional, Sequence

import torch
import numpy as np

from .tabular_time_lagged_vae import TabularTimeLaggedVAE

Array = Union[torch.Tensor, np.array, Sequence[float]]


class TabularConditionalTimeLaggedVAE(TabularTimeLaggedVAE):
    def __init__(
        self,
        x_dim: int,
        c_dim: int,
        latent_dim: int,
        hidden_layers: Sequence[int],
        x_label: str,
        xhat_label: str,
        optimizer: str = "Adam",
        lr: float = 1e-3,
        beta: float = 1.0,
        prior_mode: str = "isotropic",
        prior_logvar: Optional[Array] = None,
        learn_prior_logvar: bool = False,
        autocorr: bool = False,
        alpha: Optional[float] = 1,
    ):

        super().__init__(
            x_dim=x_dim,
            latent_dim=latent_dim,
            hidden_layers=hidden_layers,
            x_label=x_label,
            xhat_label=xhat_label,
            optimizer=optimizer,
            lr=lr,
            beta=beta,
            prior_mode=prior_mode,
            prior_logvar=prior_logvar,
            learn_prior_logvar=learn_prior_logvar,
            autocorr=autocorr,
            c_dim=c_dim,
            alpha=alpha,
        )

        self.c_label = "condition"

    def parse_batch(self, batch, parse_mask):
        x = batch[self.hparams.x_label].float()
        target = batch[self.xhat_label].float()
        condition = batch[self.c_label].float()
        return x, {"target": target, "condition": condition}

    def _step(self, stage, batch, batch_idx, logger, parse_mask=None):

        x = self.parse_batch(batch, parse_mask)
        if isinstance(x, tuple):
            x, forward_kwargs = x
            condition = forward_kwargs["condition"]
        else:
            forward_kwargs = dict()

        (
            _,
            mu,
            _,
            loss,
            recon_loss,
            kld_loss,
            autocorr_loss,
            rcl_per_input_dimension,
            kld_per_lt_dimension,
        ) = self.forward(x, **forward_kwargs)

        self.log(f"{stage} reconstruction loss", recon_loss, logger=logger)
        self.log(f"{stage} kld loss", kld_loss, logger=logger)
        self.log(f"{stage}_loss", loss, logger=logger)

        if torch.is_tensor(autocorr_loss):
            if autocorr_loss.requires_grad:
                autocorr_loss = autocorr_loss.detach()

        results = {
            "loss": loss,
            f"{stage}_loss": loss.detach(),  # for epoch end logging purposes
            "recon_loss": recon_loss.detach(),
            "kld_loss": kld_loss.detach(),
            "autocorr_loss": autocorr_loss,
            "batch_idx": batch_idx,
        }

        if stage == "test":
            results.update(
                {
                    "mu": mu.detach(),
                    "kld_per_latent_dimension": kld_per_lt_dimension.detach().float(),
                    "rcl_per_input_dimension": rcl_per_input_dimension.detach().float(),
                    "condition": condition.detach().float(),
                }
            )

        return results
