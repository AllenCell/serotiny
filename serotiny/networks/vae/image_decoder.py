import numpy as np

import torch.nn as nn

from serotiny.networks import BasicCNN


class ImageDecoder(nn.Module):
    def __init__(
        self,
        compressed_img_shape,
        hidden_channels,
        intermediate_sizes,
        latent_dim,
        output_dims,
        output_channels,
        mode,
        non_linearity,
        skip_connections,
        batch_norm,
        kernel_size,
    ):
        super().__init__()

        self.compressed_img_shape = compressed_img_shape
        compressed_img_size = np.prod(compressed_img_shape) * hidden_channels[0]
        orig_img_size = np.prod(output_dims)

        hidden_channels[-1] = output_channels
        self.hidden_channels = hidden_channels
        self.linear_decompress = nn.Linear(latent_dim, compressed_img_size)

        self.deconv = BasicCNN(
            hidden_channels[0],
            output_dim=orig_img_size,
            hidden_channels=hidden_channels,
            input_dims=compressed_img_shape,
            upsample_layers={
                i: tuple(size) for (i, size) in enumerate(intermediate_sizes)
            },
            up_conv=True,
            flat_output=False,
            kernel_size=kernel_size,
            mode=mode,
            non_linearity=non_linearity,
            skip_connections=skip_connections,
            batch_norm=batch_norm,
        )

    def forward(self, z):
        z = self.linear_decompress(z)
        z = z.view(
            z.shape[0],  # batch size
            self.hidden_channels[0],
            *self.compressed_img_shape
        )

        z = self.deconv(z)
        z = z.clamp(max=50)

        return z
