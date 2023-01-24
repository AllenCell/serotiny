import numpy as np

import torch.nn as nn
import torch


class ImageDecoder(nn.Module):
    def __init__(
        self,
        encoder,
        deconv_block,
        input_dims,
        in_channels,
        latent_dim,
    ):
        super().__init__()
        inp = torch.zeros(1, in_channels, *input_dims)
        compressed_img_shape = encoder(inp).shape
        self.compressed_img_shape = compressed_img_shape[2:]

        deconv_layers = []

        if isinstance(encoder[0][0], torch.nn.Conv3d):
            mode = "3d"
        elif isinstance(encoder[0][0], torch.nn.Conv2d):
            mode = "2d"

        for index in range(len(encoder) - 1, -1, -1):
            conv = encoder[index][0]
            if index == len(encoder) - 1:
                self.first_out_channels = conv.out_channels
                compressed_img_size = (
                    np.prod(self.compressed_img_shape) * self.first_out_channels
                )
            if index == 0:
                out_channels = in_channels
            else:
                out_channels = conv.out_channels

            if mode == "3d":
                deconv_layers.append(
                    torch.nn.LazyConvTranspose3d(
                        out_channels=out_channels,
                        kernel_size=conv.kernel_size,
                        stride=conv.stride,
                    )
                )

            else:
                deconv_layers.append(
                    torch.nn.LazyConvTranspose2d(
                        out_channels=out_channels,
                        kernel_size=conv.kernel_size,
                        stride=conv.stride,
                    )
                )
            if index != 0:
                deconv_layers.append(deconv_block)

        deconv = nn.Sequential(*deconv_layers)
        self.deconv = deconv
        self.linear_decompress = nn.Linear(latent_dim, compressed_img_size)

    def forward(self, z):
        z = self.linear_decompress(z)
        z = z.view(
            z.shape[0],  # batch size
            self.first_out_channels,
            *self.compressed_img_shape
        )
        z = self.deconv(z)
        z = z.clamp(max=50)
        return z
