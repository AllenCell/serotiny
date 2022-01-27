from typing import Union, Optional, Sequence

import torch
import numpy as np

from serotiny.networks.mlp import MLP
from .base_vae import BaseVAE

Array = Union[torch.Tensor, np.array, Sequence[float]]


class TabularVAE(BaseVAE):
    def __init__(
        self,
        x_dim: int,
        latent_dim: int,
        hidden_layers: Sequence[int],
        x_label: str,
        loss_mask_label: Optional[str] = None,
        optimizer = torch.optim.Adam,
        beta: float = 1.0,
        prior_mode: str = "isotropic",
        prior_logvar: Optional[Array] = None,
        learn_prior_logvar: bool = False,
    ):
        encoder = MLP(
            x_dim,
            2 * latent_dim,
            hidden_layers=hidden_layers,
        )

        decoder = MLP(
            latent_dim,
            x_dim,
            hidden_layers=hidden_layers,
        )

        super().__init__(
            encoder=encoder,
            decoder=decoder,
            latent_dim=latent_dim,
            optimizer=optimizer,
            x_label=x_label,
            loss_mask_label=loss_mask_label,
            lr=lr,
            beta=beta,
            prior_mode=prior_mode,
            prior_logvar=prior_logvar,
            learn_prior_logvar=learn_prior_logvar,
        )
