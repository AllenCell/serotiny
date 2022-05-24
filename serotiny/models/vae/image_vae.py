import logging
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss as Loss

from serotiny.networks import BasicCNN
from serotiny.networks.vae import ImageDecoder
from serotiny.networks.utils import weight_init

from .base_vae import BaseVAE
from .priors import Prior

Array = Union[torch.Tensor, np.ndarray, Sequence[float]]
logger = logging.getLogger("lightning")
logger.propagate = False


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
        reconstruction_reduce: str = "mean",
        skip_connections: bool = True,
        batch_norm: bool = True,
        mode: str = "3d",
        priors: Optional[Sequence[Prior]] = None,
        kernel_size: int = 3,
        cache_outputs: Sequence = ("test",),
        final_non_linearity: nn.Module = nn.Threshold(6, 6),
    ):

        encoder = BasicCNN(
            in_channels=in_channels,
            input_dims=input_dims,
            output_dim=latent_dim * 2,  # because it has to return mu and sigma
            hidden_channels=hidden_channels,
            max_pool_layers=max_pool_layers,
            mode=mode,
            kernel_size=kernel_size,
            non_linearity=non_linearity,
            skip_connections=skip_connections,
            batch_norm=batch_norm,
            final_non_linearity=final_non_linearity,
        )
        encoder.apply(weight_init)
        nn.utils.spectral_norm(encoder.output)

        dummy_out, intermediate_sizes = encoder.conv_forward(
            torch.zeros(1, in_channels, *input_dims), return_sizes=True
        )

        compressed_img_shape = dummy_out.shape[2:]

        intermediate_sizes = [input_dims] + intermediate_sizes[:-1]
        intermediate_sizes = intermediate_sizes[::-1]

        decoder = ImageDecoder(
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

        super().__init__(
            encoder=encoder,
            decoder=decoder,
            latent_dim=latent_dim,
            optimizer=optimizer,
            x_label=x_label,
            id_label=id_label,
            loss_mask_label=loss_mask_label,
            beta=beta,
            reconstruction_loss=reconstruction_loss,
            reconstruction_reduce=reconstruction_reduce,
            priors=priors,
            cache_outputs=cache_outputs,
        )
