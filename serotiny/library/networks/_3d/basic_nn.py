"""
Basic Neural Net for 3D classification
"""

from torch import nn


def _conv_layer(in_c, out_c, kernel_size=(3, 3, 3), padding=0):
    return nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=kernel_size, padding=0),
        nn.ReLU(),
        nn.BatchNorm3d(out_c),
    )


class BasicCNN_3D(nn.Module):
    def __init__(self, in_channels=6, num_classes=5, num_layers=5, max_pool_layers={0, 1, 2}, dimensions=None):
        super().__init__()
        self.network_name = "BasicNeuralNetwork_3D"
        self.num_classes = num_classes
        self.mp = nn.MaxPool3d(kernel_size=2, padding=0)
        self.max_pool_layers = max_pool_layers

        layers = []

        out_channels = 4
        for i in range(num_layers):
            layers.append(
               _conv_layer(in_channels, out_channels, kernel_size=(3, 3, 3))
            )
            in_channels = out_channels
            out_channels = out_channels * int(1.5**(i))

        self.layers = nn.Sequential(*layers)

        # hard-coded here because I'm also fixing the input dims elsewhere
        # TODO: change this
        self.output = nn.Linear(2880, num_classes)

    def forward(self, x):
        print(x.shape)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.max_pool_layers:
                x = self.mp(x)

        print(x.shape)

        return self.output(x.view(x.shape[0], -1))
