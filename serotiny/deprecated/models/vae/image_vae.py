import logging
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss as Loss

from serotiny.networks import BasicCNN
from serotiny.networks.vae import ImageDecoderBasicCNN
from serotiny.networks.utils import weight_init
import serotiny

from .base_vae import BaseVAE
from .priors import Prior

Array = Union[torch.Tensor, np.ndarray, Sequence[float]]
logger = logging.getLogger("lightning")
logger.propagate = False
from omegaconf import DictConfig


class ImageVAE(BaseVAE):
    def __init__(
        self,
        latent_dim: Union[int, Sequence[int]],
        in_channels: int,
        hidden_channels: Sequence[int],
        max_pool_layers: Sequence[int],
        input_dims: Sequence[int],
        x_label: str,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        beta: float = 1.0,
        id_label: Optional[str] = None,
        non_linearity: Optional[nn.Module] = None,
        decoder_non_linearity: Optional[nn.Module] = None,
        loss_mask_label: Optional[str] = None,
        reconstruction_loss: Loss = nn.MSELoss(reduction="none"),
        skip_connections: bool = True,
        batch_norm: bool = True,
        mode: str = "3d",
        prior: Optional[Sequence[Prior]] = None,
        kernel_size: int = 3,
        cache_outputs: Sequence = ("test",),
        encoder_clamp: Optional[int] = 6,
        # final_non_linearity: Optional[nn.Module] = None,
    ):
        if isinstance(
            prior[x_label], serotiny.models.vae.priors.IsotropicGaussianPrior
        ):
            encoder_latent_dims = 2 * latent_dim
        else:
            encoder_latent_dims = latent_dim

        encoder = BasicCNN(
            in_channels=in_channels,
            input_dims=input_dims,
            output_dim=encoder_latent_dims,  # because it has to return mu and sigma
            hidden_channels=hidden_channels,
            max_pool_layers=max_pool_layers,
            mode=mode,
            kernel_size=kernel_size,
            non_linearity=non_linearity,
            skip_connections=skip_connections,
            batch_norm=batch_norm,
            encoder_clamp=encoder_clamp,
        )
        encoder.apply(weight_init)

        dummy_out, intermediate_sizes = encoder.conv_forward(
            torch.zeros(1, in_channels, *input_dims), return_sizes=True
        )

        compressed_img_shape = dummy_out.shape[2:]

        intermediate_sizes = [input_dims] + intermediate_sizes[:-1]
        intermediate_sizes = intermediate_sizes[::-1]

        decoder = ImageDecoderBasicCNN(
            compressed_img_shape=compressed_img_shape,
            hidden_channels=list(reversed(hidden_channels)),
            intermediate_sizes=intermediate_sizes,
            latent_dim=latent_dim,
            output_dims=tuple(input_dims),
            output_channels=in_channels,
            mode=mode,
            non_linearity=non_linearity,
            skip_connections=skip_connections,
            batch_norm=batch_norm,
            kernel_size=kernel_size,
        )
        decoder.apply(weight_init)
        nn.utils.spectral_norm(decoder.linear_decompress)

        if decoder_non_linearity is not None:
            decoder = nn.Sequential(decoder, decoder_non_linearity)

        if not isinstance(encoder, (dict, DictConfig)):
            assert x_label is not None
            encoder = {x_label: encoder}

        if not isinstance(decoder, (dict, DictConfig)):
            assert x_label is not None
            decoder = {x_label: decoder}

        if not isinstance(reconstruction_loss, (dict, DictConfig)):
            assert x_label is not None
            reconstruction_loss = {x_label: reconstruction_loss}

        if not isinstance(prior, (dict, DictConfig)):
            assert x_label is not None
            prior = {x_label: prior}

        super().__init__(
            encoder=encoder,
            decoder=decoder,
            latent_dim=latent_dim,
            optimizer=optimizer,
            x_label=x_label,
            id_label=id_label,
            beta=beta,
            reconstruction_loss=reconstruction_loss,
            cache_outputs=cache_outputs,
            optimizier=optimizer,
            prior=prior,
        )
