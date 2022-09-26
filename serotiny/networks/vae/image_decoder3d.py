import numpy as np

import torch.nn as nn
import torch


class ImageDecoder3D(nn.Module):
    def __init__(
        self, encoder, input_dims, in_channels, latent_dim,
    ):
        super().__init__()
        inp = torch.zeros(1, in_channels, *input_dims)
        compressed_img_shape = encoder(inp).shape
        self.compressed_img_shape = compressed_img_shape[2:]

        orig_img_size = np.prod(input_dims)
        deconv_layers = []
        encoder = encoder[0]
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

            deconv_layers.append(
                torch.nn.LazyConvTranspose3d(
                    out_channels=out_channels,
                    kernel_size=conv.kernel_size,
                    stride=conv.stride,
                )
            )
            deconv_layers.append(torch.nn.LeakyReLU())
            deconv_layers.append(torch.nn.LazyBatchNorm3d())

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
