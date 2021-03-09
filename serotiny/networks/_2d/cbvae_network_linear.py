import torch

from torch import nn

from ..weight_init import weight_init


class CBVAENetworkLinear(nn.Module):
    def __init__(
        self,
        x_dim=295,
        c_dim=25,
        enc_layers=[295, 256, 256, 256, 256, 256, 512, 512],
        dec_layers=[512, 512, 256, 256, 256, 256, 256, 295],
    ):
        # super(Enc, self).__init__()
        super().__init__()

        self.xdim = x_dim
        self.cdim = c_dim
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
        # decoder part
        decoder_layers = []
        # decoder_net = nn.Sequential()
        for j, (i, k) in enumerate(zip(dec_layers[0::1], dec_layers[1::1])):
            if j == 0:
                decoder_layers.append(nn.Linear(i + self.cdim, k))
                decoder_layers.append(nn.BatchNorm1d(k))
                decoder_layers.append(nn.ReLU())
            elif j == len(dec_layers) - 2:
                decoder_layers.append(nn.Linear(i, k))
            else:
                decoder_layers.append(nn.Linear(i, k))
                decoder_layers.append(nn.BatchNorm1d(k))
                decoder_layers.append(nn.ReLU())

        self.decoder_net = nn.Sequential(*decoder_layers)
        self.decoder_net.apply(weight_init)

    def encoder(self, x, c):

        concat_input = torch.cat([x, c], 1)
        h = self.encoder_net(concat_input)
        return self.fc1(h), self.fc2(h)

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu)  # return z sample

    def decoder(self, z, c):
        concat_input = torch.cat([z, c], 1)
        return self.decoder_net(concat_input)

    def forward(self, x, c):
        mu, log_var = self.encoder(x, c)
        z = self.sampling(mu, log_var)
        return self.decoder(z, c), mu, log_var
