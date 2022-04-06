from typing import Optional, Sequence

import torch
import torch.nn as nn

# from pytorch3d.transforms.rotation_conversions import euler_angles_to_matrix

from serotiny.networks.layers import conv_block


def make_parallelepipeds(input_dims, batch_size=0):

    ranges = [torch.linspace(-1, 1, dim) for dim in input_dims]

    ppiped = torch.stack(torch.meshgrid(*ranges))

    if batch_size:
        repeats = (len(input_dims) + 1) * [1]
        return ppiped.unsqueeze(0).repeat(batch_size, *repeats)
    return ppiped


class ImplicitDecoder(nn.Module):
    def __init__(
        self,
        latent_dims,
        hidden_channels: Sequence[int],
        non_linearity: Optional[nn.Module] = None,
        final_non_linearity: Optional[nn.Module] = None,
        mode: str = "3d",
    ):
        super().__init__()
        self.mode = mode
        self.latent_dims = latent_dims
        _mode = 3 if mode == "3d" else 2

        if non_linearity is None:
            non_linearity = nn.ReLU()

        if final_non_linearity is None:
            self.final_non_linearity = nn.SiLU()

        layers = []
        _in_channels = latent_dims
        for ix, out_channels in enumerate(hidden_channels):
            layers.append(
                conv_block(
                    _in_channels + (_mode if ix == 0 else 0),
                    out_channels,
                    kernel_size=1,
                    up_conv=False,
                    non_linearity=non_linearity,
                    mode=mode,
                )
            )
            _in_channels = out_channels

        self.layers = nn.ModuleList(layers)

    def forward(self, x, input_dims=(10, 10, 10), angles=None, translations=None):
        ppipeds = make_parallelepipeds(input_dims, batch_size=x.shape[0])

        if angles is not None:
            pass
            #  rot_matrices = euler_angles_to_matrix(angles, convention="XYZ")
            #  ppipeds = torch.matmul(ppipeds, rot_matrices)

        if translations is not None:
            pass
            #  ppipeds = ppipeds + translations.view(
            #      x.shape[0], 1, *([1] * len(input_dims))
            #  )

        ppipeds = ppipeds.to(x.device)

        x = x.view(*x.shape, *([1] * len(input_dims)))
        x = x.expand(-1, -1, *input_dims)

        x = torch.cat((ppipeds, x), axis=1)
        for layer in self.layers:
            res = layer(x)
            if res.shape[1] == x.shape[1]:
                x = res + x
            else:
                x = res
        return self.final_non_linearity(x)