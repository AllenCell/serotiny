import torch

from torch import nn

from ..weight_init import weight_init

class ResidualLayer(nn.Module):
    def __init__(self, n_in, n_out, n_bottleneck=32, bn=True, activation="relu"):
        super(ResidualLayer, self).__init__()

        self.n_in = n_in
        self.n_out = n_out

        self.activation = nn.ReLU()

        main_layer = []
        main_layer.append(nn.Linear(n_in, n_bottleneck))
        if bn:
            main_layer.append(nn.BatchNorm1d(n_bottleneck))

        main_layer.append(nn.ReLU())
        main_layer.append(nn.Linear(n_bottleneck, n_out))

        if bn:
            main_layer.append(nn.BatchNorm1d(num_features=n_out))

        self.residual_block = nn.Sequential(*main_layer)

        if n_in != n_out:
            self.bypass = nn.Linear(n_in, n_out)
        else:
            self.bypass = nn.Sequential()

    def forward(self, x):
        x_out = self.residual_block(x) + self.bypass(x)
        x_out = self.activation(x_out)
        return x_out

class CBVAEEncoderMLPResnet(nn.Module):
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
                encoder_layers.append(ResidualLayer(i + self.cdim, k))
            #     encoder_layers.append(nn.ReLU())
            elif j == len(enc_layers) - 2:
                self.fc1 = ResidualLayer(i, k, activation=None)
                self.fc2 = ResidualLayer(i, k, activation=None)
            else:
                encoder_layers.append(ResidualLayer(i, k))

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
