from typing import Union, Sequence, Dict
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss as Loss

from serotiny.utils import init
from .conditional_vae import ConditionalVAE

Array = Union[torch.Tensor, np.array, Sequence[float]]
logger = logging.getLogger("lightning")
logger.propagate = False


class ConditionalPriorVAE(ConditionalVAE):
    def __init__(
        self,
        encoder: Union[nn.Module, Dict],
        decoder: Union[nn.Module, Dict],
        latent_dim: Union[int, Sequence[int]],
        prior_encoder: Union[nn.Module, Dict],
        optimizer: str,
        lr: float,
        x_label: str,
        c_label: Union[str, int, Sequence[int]],
        loss_mask_label: Optional[str] = None,
        recon_loss: Union[Loss, str] = torch.nn.MSELoss,
        condition_mode: str = "label",
    ):

        super().__init__(
            encoder=encoder,
            decoder=decoder,
            latent_dim=latent_dim,
            optimizer=optimizer,
            lr=lr,
            beta=1,
            x_label=x_label,
            c_label=c_label,
            loss_mask_label=loss_mask_label,
            recon_loss=recon_loss,
            condition_mode=condition_mode,
            prior_mode="anisotropic",
            prior_logvar=None,
            learn_prior_logvar=False,
        )

        if isinstance(prior_encoder, dict):
            prior_encoder = init(prior_encoder)
        self.prior_encoder = prior_encoder

    def forward(self, x, condition):
        self.prior_logvar = self.prior_encoder(condition)
        return super().forward(x)
