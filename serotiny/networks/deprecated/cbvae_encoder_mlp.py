import torch

from torch import nn

from ..weight_init import weight_init


class CBVAEEncoderMLP(nn.Module):
    def __init__(
        self,
        x_dim=295,
        c_dim=25,
        hidden_layers=[256, 256, 256, 256],
        latent_dims=64,
    ):
        # super(Enc, self).__init__()
        super().__init__()

        self.xdim = x_dim
        self.cdim = c_dim
        self.hidden_layers = hidden_layers
        self.latent_dims = latent_dims

        self.enc_layers = self.hidden_layers.copy()
        self.enc_layers.insert(0, self.xdim)
        self.enc_layers.append(self.latent_dims)
        # encoder part

        encoder_layers = []
        # encoder_net = nn.Sequential()
        for j, (i, k) in enumerate(zip(self.enc_layers[0::1], self.enc_layers[1::1])):
            if j == 0:
                encoder_layers.append(nn.Linear(i + self.cdim, k))
                encoder_layers.append(nn.BatchNorm1d(k))
                encoder_layers.append(nn.ReLU())
            elif j == len(self.enc_layers) - 2:
                self.fc1 = nn.Linear(self.enc_layers[-2], self.enc_layers[-1])
                self.fc2 = nn.Linear(self.enc_layers[-2], self.enc_layers[-1])
            else:
                encoder_layers.append(nn.Linear(i, k))
                encoder_layers.append(nn.BatchNorm1d(k))
                encoder_layers.append(nn.ReLU())

        self.encoder_net = nn.Sequential(*encoder_layers)
        self.encoder_net.apply(weight_init)

    def encoder(self, x, c):

        concat_input = torch.cat([x, c], 1)
        h = self.encoder_net(concat_input)
        return self.fc1(h), self.fc2(h)

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu)  # return z sample

    def forward(self, x, c):
        mu, log_var = self.encoder(x, c)
        z = self.sampling(mu, log_var)
        return mu, log_var, z
