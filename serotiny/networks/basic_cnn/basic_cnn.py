import logging
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn

from serotiny.networks.layers import ConvBlock, spatial_pyramid_pool

log = logging.getLogger(__name__)


class BasicCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        output_dim: int,
        hidden_channels: Sequence[int],
        kernel_size: int = 3,
        stride: int = 1,
        input_dims: Optional[Sequence[int]] = None,
        max_pool_layers: Sequence[int] = [],
        upsample_layers: Sequence[int] = [],
        pyramid_pool_splits: Optional[Sequence[int]] = None,
        flat_output: bool = True,
        up_conv: bool = False,
        non_linearity: Optional[nn.Module] = None,
        final_non_linearity: Optional[nn.Module] = None,
        skip_connections: Union[bool, Sequence[int]] = False,
        batch_norm: bool = True,
        mode: str = "3d",
    ):
        """Instantiate a 3D CNN.

        Parameters
        ----------
        in_channels: int
            Number of input channels
        input_dims: Sequence[int]
            Dimensions of input channels
        output_dim: int
            Dimensionality of the output
        hidden_channels: Sequence[int]
            Number of channels for each hidden layer. (And implicitly, the
            depth of the network)
        max_pool_layers: int
            Sequence of layers in which to apply a max pooling operation
        """
        super().__init__()
        self.output_dim = output_dim
        self.mode = mode
        _mode = 3 if mode == "3d" else 2

        max_pool = nn.MaxPool3d if mode == "3d" else nn.MaxPool2d
        self.max_pool = max_pool(kernel_size=2, padding=0)
        self.max_pool_layers = max_pool_layers
        self.upsample_layers = upsample_layers
        self.pyramid_pool_splits = pyramid_pool_splits
        self.flat_output = flat_output

        if isinstance(skip_connections, (tuple, list)):
            self.skip_connections = skip_connections
        else:
            if skip_connections:
                self.skip_connections = tuple(range(len(hidden_channels)))
            else:
                self.skip_connections = tuple()

        layers = []

        _in_channels = in_channels
        for ix, out_channels in enumerate(hidden_channels):
            layers.append(
                ConvBlock(
                    _in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    up_conv=up_conv,
                    non_linearity=non_linearity,
                    mode=mode,
                    skip_connection=(ix in self.skip_connections),
                    batch_norm=batch_norm,
                )
            )
            _in_channels = out_channels

        self.layers = nn.Sequential(*layers)

        # feed dummy input through convolutional part of the network
        # to infer the needed input size of the final fully connected layer
        if pyramid_pool_splits is None:
            assert input_dims is not None
            dummy_conv_output = self.conv_forward(
                torch.zeros(1, in_channels, *input_dims)
            )
            compressed_size = np.prod(dummy_conv_output.shape[1:])
        else:
            if input_dims is None:
                log.warn(
                    f"You really should define input_dims..., "
                    f"I'm assuming {[200]*_mode}"
                )
                input_dims = [200] * _mode
            dummy_conv_output = self.conv_forward(
                torch.zeros(1, in_channels, *input_dims)
            )
            dummy_compressed = spatial_pyramid_pool(
                dummy_conv_output, self.pyramid_pool_splits
            )
            compressed_size = np.prod(dummy_compressed.shape[1:])

        log.info(f"Determined 'compressed size': {compressed_size} for CNN")

        if flat_output:
            self.output = nn.Sequential(
                nn.Linear(compressed_size, output_dim),
                nn.Identity() if final_non_linearity is None else final_non_linearity,
            )

    def conv_forward(self, x, return_sizes=False):
        if return_sizes:
            sizes = [None] * len(self.layers)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.max_pool_layers:
                x = self.max_pool(x)

            if i in self.upsample_layers:
                output_size = self.upsample_layers[i]
                if isinstance(output_size, Sequence):
                    upsample = nn.Upsample(size=output_size)
                elif isinstance(output_size, (float, int)):
                    upsample = nn.Upsample(scale_factor=output_size)
                else:
                    raise TypeError

                x = upsample(x)

            if return_sizes:
                sizes[i] = x.shape[2:]

        if return_sizes:
            return x, sizes

        return x

    def forward(self, x):
        x = self.conv_forward(x)
        if self.pyramid_pool_splits is not None:
            x = spatial_pyramid_pool(x, self.pyramid_pool_splits)

        if self.flat_output:
            x = self.output(x.view(x.shape[0], -1))

        return x
