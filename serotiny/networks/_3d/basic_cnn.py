from typing import Sequence, Optional
import logging

import numpy as np
import torch
import torch.nn as nn

from serotiny.networks.layers.spatial_pyramid_pool import spatial_pyramid_pool

log = logging.getLogger(__name__)

def _conv_layer(
    in_c: int, out_c: int, kernel_size: Sequence[int] = (3, 3, 3), padding: int = 0
):
    """
    Util function to instantiate a convolutional block.

    Parameters
    ----------
    in_c: int
        number of input channels
    out_c: int
        number of output channels
    kernel_size: Sequence[int] (defaults to (3, 3, 3))
        dimensions of the convolutional kernel to be applied
    padding:
        padding value for the convolution (defaults to 0)
    """
    return nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=kernel_size, padding=padding),
        nn.ReLU(),
        nn.BatchNorm3d(out_c),
    )


class BasicCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        output_dim: int,
        hidden_channels: Sequence[int],
        input_dims: Optional[Sequence[int]] = None,
        max_pool_layers: Sequence[int] = [],
        pyramid_pool_splits: Optional[Sequence[int]] = None,
    ):
        """
        Instantiate a 3D CNN

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
        self.max_pool = nn.MaxPool3d(kernel_size=2, padding=0)
        self.max_pool_layers = max_pool_layers
        self.pyramid_pool_splits = pyramid_pool_splits

        layers = []

        _in_channels = in_channels
        for out_channels in hidden_channels:
            layers.append(
                _conv_layer(_in_channels, out_channels, kernel_size=(3, 3, 3))
            )
            _in_channels = out_channels

        self.layers = nn.Sequential(*layers)

        # feed dummy input through convolutional part of the network
        # to infer the needed input size of the final fully connected layer
        if pyramid_pool_splits is None:
            assert input_dims is not None
            dummy_conv_output = self.conv_forward(torch.zeros(1, in_channels, *input_dims))
            compressed_size = np.prod(dummy_conv_output.shape[1:])
        else:
            if input_dims is None:
                input_dims = (200, 200, 200)
            dummy_conv_output = self.conv_forward(torch.zeros(1, in_channels, *input_dims))
            dummy_compressed = spatial_pyramid_pool(dummy_conv_output, self.pyramid_pool_splits)
            compressed_size = np.prod(dummy_compressed.shape[1:])

        log.info(f"Determined 'compressed size': {compressed_size} for CNN")

        self.output = nn.Linear(compressed_size, output_dim)

    def conv_forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.max_pool_layers:
                x = self.max_pool(x)

        return x

    def forward(self, x):
        x = self.conv_forward(x)
        if self.pyramid_pool_splits is not None:
            x = spatial_pyramid_pool(x, self.pyramid_pool_splits)
        return self.output(x.view(x.shape[0], -1))
