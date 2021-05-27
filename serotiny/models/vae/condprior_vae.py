from typing import Union, Optional, Sequence, Dict
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss as Loss

from serotiny.utils import get_class_from_path
from .conditional_vae import ConditionalVAE

Array = Union[torch.Tensor, np.array, Sequence[float]]
logger = logging.getLogger("lightning")
logger.propagate = False


class ConditionalPriorVAE(ConditionalVAE):
    def __init__(
        self,
        encoder: Union[nn.Module, str],
        decoder: Union[nn.Module, str],
        latent_dim: Union[int, Sequence[int]],
        prior_encoder: Union[nn.Module, str],
        optimizer: str,
        lr: float,
        x_label: str,
        c_label: Union[str, int, Sequence[int]],
        recon_loss: Union[Loss, str] = torch.nn.MSELoss,
        condition_mode: str = "label",
        encoder_config: Optional[Dict] = None,
        decoder_config: Optional[Dict] = None,
        prior_encoder_config: Optional[Dict] = None,
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
            recon_loss=recon_loss,
            condition_mode=condition_mode,
            prior_mode="anisotropic",
            prior_logvar=None,
            learn_prior_logvar=False,
            encoder_config=encoder_config,
            decoder_config=decoder_config
        )

        if isinstance(prior_encoder, str):
            prior_encoder = get_class_from_path(prior_encoder)
            prior_encoder = prior_encoder(**prior_encoder_config)
        self.prior_encoder = prior_encoder

    def forward(self, x, condition):
        self.prior_logvar = self.prior_encoder(condition)
        return super().forward(x)
