import math

import torch
import torch.nn as nn


def spatial_pyramid_pool(x, out_pool_sizes):

    input_dims = x.shape[2:]
    batch_size = x.shape[0]

    for i in range(len(out_pool_sizes)):
        wids = []
        pads = []

        for dim in input_dims:
            wid = int(math.ceil(dim / out_pool_sizes[i]))
            wids.append(wid)
            pad = int(math.floor((wid * out_pool_sizes[i] - dim + 1) / 2))
            pads.append(pad)

        if len(input_dims) == 1:
            maxpool = nn.MaxPool1d
        elif len(input_dims) == 2:
            maxpool = nn.MaxPool2d
        elif len(input_dims) == 3:
            maxpool = nn.MaxPool3d
        else:
            raise ValueError("Spatial Pyramid Pool only supports 1, 2, or 3d data")
        maxpool = maxpool(tuple(wids), stride=tuple(wids), padding=tuple(pads))

        out = maxpool(x)
        if i == 0:
            spp = out.view(batch_size, -1)
        else:
            spp = torch.cat((spp, out.view(batch_size, -1)), 1)

    return spp


class SpatialPyramidPool(nn.Module):
    def __init__(self, out_pool_sizes):
        super().__init__()
        self.out_pool_sizes = out_pool_sizes

    def forward(self, x):
        return spatial_pyramid_pool(x, self.out_pool_sizes)
