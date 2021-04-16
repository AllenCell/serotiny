import torch

from torch import nn

from ..weight_init import weight_init


class CBVAEDecoderMLP(nn.Module):
    def __init__(
        self,
        x_dim=295,
        c_dim=25,
        hidden_layers=[256, 256, 256, 256, 256],
        latent_dims=64,
    ):
        # super(Enc, self).__init__()
        super().__init__()

        self.xdim = x_dim
        self.cdim = c_dim
        self.hidden_layers = hidden_layers
        self.latent_dims = latent_dims

        self.dec_layers = self.hidden_layers.copy()

        self.dec_layers.insert(0, self.latent_dims)
        self.dec_layers.append(self.xdim)

        # decoder part
        decoder_layers = []
        # decoder_net = nn.Sequential()
        for j, (i, k) in enumerate(zip(self.dec_layers[0::1], self.dec_layers[1::1])):
            if j == 0:
                decoder_layers.append(nn.Linear(i + self.cdim, k))
                decoder_layers.append(nn.BatchNorm1d(k))
                decoder_layers.append(nn.ReLU())
            elif j == len(self.dec_layers) - 2:
                decoder_layers.append(nn.Linear(i, k))
            else:
                decoder_layers.append(nn.Linear(i, k))
                decoder_layers.append(nn.BatchNorm1d(k))
                decoder_layers.append(nn.ReLU())

        self.decoder_net = nn.Sequential(*decoder_layers)
        self.decoder_net.apply(weight_init)

    def decoder(self, z, c):
        concat_input = torch.cat([z, c], 1)
        return self.decoder_net(concat_input)

    def forward(self, z, c):
        return self.decoder(z, c)
