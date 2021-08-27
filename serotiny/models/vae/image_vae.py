from typing import Union, Optional, Sequence, Dict
import logging

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss as Loss
import numpy as np

from serotiny.networks._3d import BasicCNN
from serotiny.networks.weight_init import weight_init
from serotiny.utils.dynamic_imports import load_config
from .base_vae import BaseVAE

Array = Union[torch.Tensor, np.array, Sequence[float]]
logger = logging.getLogger("lightning")
logger.propagate = False

class _ImageVAEDecoder(nn.Module):
    def __init__(
        self,
        compressed_img_shape,
        hidden_channels,
        intermediate_sizes,
        latent_dim,
        output_dims,
        output_channels,
        mode
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
            upsample_layers={i:size for (i,size) in enumerate(intermediate_sizes)},
            up_conv=True,
            flat_output=False,
            mode=mode,
        )

    def forward(self, z):
        z = self.linear_decompress(z)
        z = z.view(
            z.shape[0],  # batch size
            self.hidden_channels[0],
            *self.compressed_img_shape
        )

        z = self.deconv(z)
        z = z.clamp(max=50)

        return z


class ImageVAE(BaseVAE):
    def __init__(
        self,
        latent_dim: Union[int, Sequence[int]],
        in_channels: int,
        hidden_channels: Sequence[int],
        max_pool_layers: Sequence[int],
        input_dims: Sequence[int],
        x_label: str,
        optimizer: str = "Adam",
        lr: float = 1e-3,
        beta: float = 1.0,
        decoder_non_linearity: Optional[Union[nn.Module, Dict]] = None,
        loss_mask_label: Optional[str] = None,
        recon_loss: Union[Loss, Dict] = nn.MSELoss,
        recon_reduce: str = "mean",
        prior_mode: str = "isotropic",
        prior_logvar: Optional[Array] = None,
        learn_prior_logvar: bool = False,
        mode: str = "3d",
    ):

        encoder = BasicCNN(
            in_channels=in_channels,
            input_dims=input_dims,
            output_dim=latent_dim * 2,  # because it has to return mu and sigma
            hidden_channels=hidden_channels,
            max_pool_layers=max_pool_layers,
            mode=mode,
        )
        encoder.apply(weight_init)
        nn.utils.spectral_norm(encoder.output)

        dummy_out, intermediate_sizes = encoder.conv_forward(
            torch.zeros(1, in_channels, *input_dims),
            return_sizes=True
        )

        compressed_img_shape = dummy_out.shape[2:]

        intermediate_sizes = [input_dims] + intermediate_sizes[:-1]
        intermediate_sizes = intermediate_sizes[::-1]

        decoder = _ImageVAEDecoder(
            compressed_img_shape=compressed_img_shape,
            hidden_channels=list(reversed(hidden_channels)),
            intermediate_sizes=intermediate_sizes,
            latent_dim=latent_dim,
            output_dims=input_dims,
            output_channels=in_channels,
            mode=mode
        )
        decoder.apply(weight_init)
        nn.utils.spectral_norm(decoder.linear_decompress)

        if decoder_non_linearity is not None:
            if isinstance(decoder_non_linearity, dict):
                decoder_non_linearity = load_config(decoder_non_linearity)
            decoder = nn.Sequential(decoder, decoder_non_linearity)

        super().__init__(
            encoder=encoder,
            decoder=decoder,
            latent_dim=latent_dim,
            optimizer=optimizer,
            x_label=x_label,
            loss_mask_label=loss_mask_label,
            lr=lr,
            beta=beta,
            recon_loss=recon_loss,
            recon_reduce=recon_reduce,
            prior_mode=prior_mode,
            prior_logvar=prior_logvar,
            learn_prior_logvar=learn_prior_logvar,
        )
