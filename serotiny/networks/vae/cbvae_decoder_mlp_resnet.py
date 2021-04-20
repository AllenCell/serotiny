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

class CBVAEDecoderMLPResnet(nn.Module):
    def __init__(
        self,
        x_dim=295,
        c_dim=25,
        dec_layers=[512, 512, 256, 256, 256, 256, 256, 295],
    ):
        # super(Enc, self).__init__()
        super().__init__()

        self.xdim = x_dim
        self.cdim = c_dim
        self.dec_layers = dec_layers

        # decoder part
        decoder_layers = []
        # decoder_net = nn.Sequential()
        for j, (i, k) in enumerate(zip(dec_layers[0::1], dec_layers[1::1])):
            if j == 0:
                decoder_layers.append(ResidualLayer(i + self.cdim, k))
                # decoder_layers.append(nn.ReLU())
            elif j == len(dec_layers) - 2:
                decoder_layers.append(ResidualLayer(i, k, activation=None))
            else:
                decoder_layers.append(ResidualLayer(i, k))

        self.decoder_net = nn.Sequential(*decoder_layers)
        self.decoder_net.apply(weight_init)

    def decoder(self, z, c):
        concat_input = torch.cat([z, c], 1)
        return self.decoder_net(concat_input)

    def forward(self, z, c):
        return self.decoder(z, c)
