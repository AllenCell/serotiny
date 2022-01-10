from typing import Union, Optional, Sequence

import torch
import numpy as np

from serotiny.networks.mlp import MLP
from .base_vae import BaseVAE
from serotiny.losses.elbo import calculate_elbo

Array = Union[torch.Tensor, np.array, Sequence[float]]


class TabularTimeLaggedVAE(BaseVAE):
    def __init__(
        self,
        x_dim: int,
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
        c_dim: Optional[int] = 0,
        alpha: Optional[float] = 1,
    ):
        encoder = MLP(x_dim + c_dim, 2 * latent_dim, hidden_layers=hidden_layers,)

        decoder = MLP(latent_dim + c_dim, x_dim, hidden_layers=hidden_layers,)

        super().__init__(
            encoder=encoder,
            decoder=decoder,
            latent_dim=latent_dim,
            optimizer=optimizer,
            x_label=x_label,
            lr=lr,
            beta=beta,
            prior_mode=prior_mode,
            prior_logvar=prior_logvar,
            learn_prior_logvar=learn_prior_logvar,
        )

        self.xhat_label = xhat_label
        self.autocorr = autocorr
        self.alpha = alpha

    def parse_batch(self, batch, parse_mask):
        x = batch[self.hparams.x_label].float()
        target = batch[self.xhat_label].float()
        return x, {"target": target}

    def _corr(self, x, y):
        x = x.reshape(-1)
        y = y.reshape(-1)
        mean_x = torch.mean(x)
        mean_y = torch.mean(y)
        xm = x.sub(mean_x.expand_as(x))
        ym = y.sub(mean_y.expand_as(y))
        r_num = xm.dot(ym)
        r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
        r_val = r_num / r_den
        return r_val

    def forward(self, x, **kwargs):
        mu_logvar = self.encoder(
            x, **{k: v for k, v in kwargs.items() if k in self.encoder_args}
        )

        mu, logvar = torch.split(mu_logvar, mu_logvar.shape[1] // 2, dim=1)
        assert mu.shape == logvar.shape

        z = self.sample_z(mu, logvar)

        x_hat = self.decoder(
            z, **{k: v for k, v in kwargs.items() if k in self.decoder_args}
        )

        mask_kwargs = {k: v for k, v in kwargs.items() if k in ["mask"]}
        target = {v for k, v in kwargs.items() if k in ["target"]}

        if len(target) > 0:
            x = target.pop()

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
            mode=self.prior_mode,
            prior_mu=(None if self.prior_mean is None else self.prior_mean.type_as(mu)),
            prior_logvar=self.prior_logvar,
            **mask_kwargs,
        )
        # Batch size = 1 means RCL is summed across batch
        # Batch size = x.shape[0] means RCL is mean across batch
        # batch_size = x.shape[0]
        batch_size = x.shape[0]
        
        # batch_size = 1
        if self.autocorr:
            mu_logvar_target = self.encoder(
                x, **{k: v for k, v in kwargs.items() if k in self.encoder_args}
            )

            mu_target, logvar_target = torch.split(mu_logvar_target, mu_logvar_target.shape[1] // 2, dim=1)
            assert mu_target.shape == logvar_target.shape
            autocorr_loss = (1 - self._corr(mu, mu_target))
        else:
            autocorr_loss = 0
        loss = loss + self.alpha * autocorr_loss

        return (
            x_hat,
            mu,
            logvar,
            loss / batch_size,
            recon_loss / batch_size,
            kld_loss / batch_size,
            autocorr_loss,
            rcl_per_input_dimension,
            kld_per_latent_dimension,
        )

    def _step(self, stage, batch, batch_idx, logger, parse_mask=None):

        x = self.parse_batch(batch, parse_mask)
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
                }
            )

        return results