from typing import Callable, Union, Optional, Sequence

import torch
import numpy as np

from serotiny.networks.mlp import MLP
from .basic_vae import BaseVAE

Array = Union[torch.Tensor, np.array, Sequence[float]]

class TabularVAE(BaseVAE):
    def __init__(
        self,
        x_dim: int,
        latent_dim: int,
        hidden_layers: Sequence[int],
        x_label: str,
        optimizer: str = "Adam",
        lr: float = 1e-3,
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
            optimizer=optimizer,
            x_label=x_label,
            lr=lr,
            beta=beta,
            prior_mode=prior_mode,
            prior_logvar=prior_logvar,
            learn_prior_logvar=learn_prior_logvar,
        )