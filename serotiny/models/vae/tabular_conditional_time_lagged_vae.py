from typing import Union, Optional, Sequence

import torch
import numpy as np

from serotiny.networks.mlp import MLP
from .base_vae import BaseVAE
from .tabular_time_lagged_vae import TabularTimeLaggedVAE
from serotiny.losses.elbo import calculate_elbo

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