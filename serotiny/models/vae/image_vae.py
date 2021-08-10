from typing import Union, Optional, Sequence, Dict
import logging

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss as Loss
import numpy as np

from serotiny.networks._3d import BasicCNN
from .base_vae import BaseVAE

Array = Union[torch.Tensor, np.array, Sequence[float]]
logger = logging.getLogger("lightning")
logger.propagate = False


class _ImageVAEDecoder(nn.Module):
    def __init__(
        self,
        compressed_img_shape,
        hidden_channels,
        latent_dim,
        output_dims,
        output_channels,
    ):
        super().__init__()

        self.compressed_img_shape = compressed_img_shape
        compressed_img_size = np.prod(compressed_img_shape) * hidden_channels[0]
        orig_img_size = np.prod(output_dims)

        hidden_channels[-1] = output_channels
        self.hidden_channels = hidden_channels
        self.linear_decompress = nn.Linear(latent_dim, compressed_img_size)

        self.deconv = BasicCNN(
            hidden_channels[0],
            output_dim=orig_img_size,
            hidden_channels=hidden_channels,
            input_dims=compressed_img_shape,
            up_conv=True,
            flat_output=False,
        )

    def forward(self, z):
        z = self.linear_decompress(z)
        z = z.view(
            z.shape[0],  # batch size
            self.hidden_channels[0],
            *self.compressed_img_shape
        )

        return self.deconv(z)


class ImageVAE(BaseVAE):
    def __init__(
        self,
        latent_dim: Union[int, Sequence[int]],
        in_channels: int,
        hidden_channels: Sequence[int],
        input_dims: Sequence[int],
        x_label: str,
        optimizer: str = "Adam",
        lr: float = 1e-3,
        beta: float = 1.0,
        recon_loss: Union[Loss, Dict] = nn.MSELoss,
        prior_mode: str = "isotropic",
        prior_logvar: Optional[Array] = None,
        learn_prior_logvar: bool = False,
    ):

        encoder = BasicCNN(
            in_channels=in_channels,
            input_dims=input_dims,
            output_dim=latent_dim * 2,  # because it has to return mu and sigma
            hidden_channels=hidden_channels,
        )

        compressed_img_shape = encoder.conv_forward(
            torch.zeros(1, in_channels, *input_dims)
        ).shape[2:]

        decoder = _ImageVAEDecoder(
            compressed_img_shape=compressed_img_shape,
            hidden_channels=list(reversed(hidden_channels)),
            latent_dim=latent_dim,
            output_dims=input_dims,
            output_channels=in_channels,
        )

        super().__init__(
            encoder=encoder,
            decoder=decoder,
            latent_dim=latent_dim,
            optimizer=optimizer,
            x_label=x_label,
            lr=lr,
            beta=beta,
            recon_loss=recon_loss,
            prior_mode=prior_mode,
            prior_logvar=prior_logvar,
            learn_prior_logvar=learn_prior_logvar,
        )
