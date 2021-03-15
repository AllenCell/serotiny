from typing import Sequence, Union

import numpy as np
import torch

from torch import nn


class PadLayer(nn.Module):
    def __init__(self, dimensionality: int, pad_dims: Union[int, Sequence[int]]):
        """
        Layer to apply symmetric padding according to what is given in pad_dims

        Parameters
        ----------
        dimensionality: int
            Whether this padding applies to 2d or 3d data
        pad_dims: Sequence[int]
            Size of padding to apply in each dimension
        """
        super(PadLayer, self).__init__()

        if dimensionality not in [2, 3]:
            raise ValueError("Dimensionality must be 2 or 3 for `PadLayer`")

        if isinstance(pad_dims, int):
            pad_dims = [pad_dims] * dimensionality

        assert len(pad_dims) == dimensionality
        self.dimensionality = dimensionality

        # for each dimension, copy the pad value twice, to apply symmetric padding
        self.pad_dims = []
        for pad in pad_dims:
            self.pad_dims.append(pad)
            self.pad_dims.append(pad)

        # reverse the list of padding values, because that's the order expected
        # by nn.functional.pad
        self.pad_dims = self.pad_dims[::-1]

    def forward(self, x):
        if np.sum(self.pad_dims) == 0:
            return x
        else:
            return nn.functional.pad(x, self.pad_dims, "constant", 0)
