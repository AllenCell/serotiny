import logging
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss as Loss

from serotiny.networks import BasicCNN
from serotiny.networks.utils import weight_init

from .base_vae import BaseVAE
from .priors import Prior

Array = Union[torch.Tensor, np.array, Sequence[float]]
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
            non_linearity=non_linearity,
            skip_connections=skip_connections,
            batch_norm=batch_norm,
            final_non_linearity=final_non_linearity,
        )
        encoder.apply(weight_init)
        nn.utils.spectral_norm(encoder.output[0])

        dummy_out, intermediate_sizes = encoder.conv_forward(
            torch.zeros(1, in_channels, *input_dims), return_sizes=True
        )

        compressed_img_shape = dummy_out.shape[2:]

        intermediate_sizes = [input_dims] + intermediate_sizes[:-1]
        intermediate_sizes = intermediate_sizes[::-1]

        decoder = _ImageVAEDecoder(
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


class _ImageVAEDecoder(nn.Module):
    def __init__(
        self,
        compressed_img_shape,
        hidden_channels,
        intermediate_sizes,
        latent_dim,
        output_dims,
        output_channels,
        mode,
        non_linearity,
        skip_connections,
        batch_norm,
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
            upsample_layers={
                i: tuple(size) for (i, size) in enumerate(intermediate_sizes)
            },
            up_conv=True,
            flat_output=False,
            mode=mode,
            non_linearity=non_linearity,
            skip_connections=skip_connections,
            batch_norm=batch_norm,
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
