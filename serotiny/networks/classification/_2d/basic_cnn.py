"""
Basic Neural Net for 2D classification
"""

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn


def _conv_layer(
    in_c: int, out_c: int, kernel_size: Sequence[int] = (3, 3), padding: int = 0
):
    """
    Util function to instantiate a convolutional block.

    Parameters
    ----------
    in_c: int
        number of input channels
    out_c: int
        number of output channels
    kernel_size: Sequence[int]
        dimensions of the convolutional kernel to be applied
    padding:
        padding value for the convolution (defaults to 0)
    """
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=padding),
        nn.ReLU(),
        nn.BatchNorm2d(out_c),
    )


class BasicCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        input_dims: Sequence[int],
        num_classes: int,
        num_layers: int = 3,
        max_pool_layers: Sequence[int] = [],
    ):
        """
        Instantiate a 2D CNN for classification

        Parameters
        ----------
        in_channels: int
            Number of input channels
        input_dims: Sequence[int]
            Dimensions of input channels
        num_classes: int
            Number of classes in the dataset
        num_layers: int
            Depth of the network
        max_pool_layers: int
            Sequence of layers in which to apply a max pooling operation
        pretrained: bool
            Flag to decide whether to train model from scratch or initialize by
            leveraging a pretrained 2d resnet
        """
        super().__init__()
        self.num_classes = num_classes
        self.max_pool = nn.MaxPool2d(kernel_size=2, padding=0)
        self.max_pool_layers = max_pool_layers

        layers = []

        out_channels = 4
        _in_channels = in_channels
        for i in range(num_layers):
            layers.append(_conv_layer(_in_channels, out_channels, kernel_size=(3, 3)))
            _in_channels = out_channels
            out_channels = out_channels * int(1.5 ** (i))

        self.layers = nn.Sequential(*layers)

        # feed dummy input through convolutional part of the network
        # to infer the needed input size of the final fully connected layer
        dummy_conv_output = self.conv_forward(torch.zeros(1, in_channels, *input_dims))
        compressed_size = np.prod(dummy_conv_output.shape[1:])

        self.output = nn.Linear(compressed_size, num_classes)

    def conv_forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.max_pool_layers:
                x = self.max_pool(x)

        return x

    def forward(self, x):
        x = self.conv_forward(x)
        return self.output(x.view(x.shape[0], -1))
