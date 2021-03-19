import torch

from torch import nn

from ..weight_init import weight_init


class CBVAEEncoderMLP(nn.Module):
    def __init__(
        self,
        x_dim=295,
        c_dim=25,
        enc_layers=[295, 256, 256, 256, 256, 256, 512, 512],
    ):
        # super(Enc, self).__init__()
        super().__init__()

        self.xdim = x_dim
        self.cdim = c_dim
        self.enc_layers = enc_layers
        # encoder part

        encoder_layers = []
        # encoder_net = nn.Sequential()
        for j, (i, k) in enumerate(zip(enc_layers[0::1], enc_layers[1::1])):
            if j == 0:
                encoder_layers.append(nn.Linear(i + self.cdim, k))
                encoder_layers.append(nn.BatchNorm1d(k))
                encoder_layers.append(nn.ReLU())
            elif j == len(enc_layers) - 2:
                self.fc1 = nn.Linear(enc_layers[-2], enc_layers[-1])
                self.fc2 = nn.Linear(enc_layers[-2], enc_layers[-1])
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
