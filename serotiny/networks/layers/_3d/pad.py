import numpy as np
import torch

from torch import nn

class PadLayer(nn.Module):
    def __init__(self, pad_dims):
        super(PadLayer, self).__init__()

        self.pad_dims = pad_dims

    def forward(self, x):
        if np.sum(self.pad_dims) == 0:
            return x
        else:
            return nn.functional.pad(
                x, [self.pad_dims[1], 0, self.pad_dims[0]], "constant", 0
            )
