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
        self._mode = 3 if mode == "3d" else 2
        self.latent_dims = latent_dims

        if non_linearity is None:
            non_linearity = nn.ReLU()

        if final_non_linearity is None:
            final_non_linearity = nn.Identity()

        self.final_non_linearity = final_non_linearity

        layers = []
        _in_channels = latent_dims
        for index, out_channels in enumerate(hidden_channels):
            layers.append(
                conv_block(
                    _in_channels + self._mode + (self.latent_dims if index != 0 else 0),
                    out_channels,
                    kernel_size=1,
                    up_conv=False,
                    non_linearity=non_linearity,
                    mode=mode,
                )
            )
            _in_channels = out_channels

        self.layers = nn.ModuleList(layers)

    def forward(self, z, input_dims=(10, 10, 10), angles=None, translations=None):
        ppipeds = make_parallelepipeds(input_dims, batch_size=z.shape[0])

        if angles is not None:
            pass
            #  rot_matrices = euler_angles_to_matrix(angles, convention="XYZ")
            #  ppipeds = torch.matmul(ppipeds, rot_matrices)

        if translations is not None:
            pass
            #  ppipeds = ppipeds + translations.view(
            #      x.shape[0], 1, *([1] * len(input_dims))
            #  )

        ppipeds = ppipeds.to(z.device)

        z = z.view(*z.shape, *([1] * len(input_dims)))
        z = z.expand(-1, -1, *input_dims)

        y = z
        # Batch,channel (lt dim),input dims
        for index, layer in enumerate(self.layers):
            to_cat = (ppipeds, y) if index == 0 else (ppipeds, z, y)
            res = layer(torch.cat(to_cat, axis=1))
            # if output and input dimensions match
            if res.shape[1] == y.shape[1]:
                # skip connection, excluding the scoordinates and the latent code
                y = res + y
            else:
                y = res

        return self.final_non_linearity(y)
