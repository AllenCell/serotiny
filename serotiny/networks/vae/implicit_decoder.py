from typing import Optional, Sequence

import torch
import torch.nn as nn

from pytorch3d.transforms.rotation_conversions import euler_angles_to_matrix

from serotiny.networks.layers import conv_block


def make_parallelepipeds(input_dims, batch_size=0):

    ranges = [
        torch.arange(-dim // 2, dim // 2 - 1 + (dim % 2 == 0))
        for dim in input_dims
    ]


    ppiped = torch.stack(torch.meshgrid(*ranges))

    if batch_size:
        repeats = (len(input_dims) + 1) * [1]
        return ppiped.unsqueeze(0).repeat(batch_size, *repeats)
    return ppiped

class ImplicitDecoder(nn.Module):
    def __init__(
        self,
        input_dims,
        in_channels: int,
        hidden_channels: Sequence[int],
        non_linearity: Optional[nn.Module] = None,
        final_non_linearity: Optional[nn.Module] = None,
        mode: str = "3d",

    ):
        super().__init__()
        self.mode = mode
        self.input_dims = input_dims
        _mode = 3 if mode == "3d" else 2

        if non_linearity is None:
            non_linearity = nn.ReLU()

        if final_non_linearity is None:
            final_non_linearity = nn.Sigmoid()

        layers = []
        _in_channels = in_channels
        for out_channels in hidden_channels:
            layers.append(
                conv_block(
                    _in_channels + len(input_dims),
                    out_channels,
                    kernel_size=1,
                    up_conv=False,
                    non_linearity=non_linearity,
                    mode=mode,
                )
            )
            _in_channels = out_channels

        if final_non_linearity is None:
            non_linearity = nn.Sigmoid()


    def forward(self, x, angles, translations, input_dims=None):
        if input_dims is None:
            input_dims = self.input_dims

        ppipeds = make_parallelepipeds(*input_dims, batch_size=x.shape[0])

        rot_matrices = euler_angles_to_matrix(angles, convention="XYZ")

        ppipeds = torch.matmul(ppipeds, rot_matrices)
        ppipeds = ppipeds + translations.view(x.shape[0], 1,
                                              *([1] * len(input_dims)))

        x = (x
             .view(*x.shape, *([1] * len(input_dims)))
             .expand(-1, -1, *input_dims)
        )

        _x = x
        for layer in self.layers:
            _x = layer(
                torch.cat((ppipeds, _x))
            )
        return self.non_linearity(_x)
