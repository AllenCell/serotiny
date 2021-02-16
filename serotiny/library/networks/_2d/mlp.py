"""
Basic Neural Net for 2D classification
"""

import torch
from torch import nn
from torch.nn import functional as F


class BasicNeuralNetwork(nn.Module):
    def __init__(self, in_channels=3, num_classes=5, dimensions=(176, 104)):
        super().__init__()
        self.network_name = "BasicNeuralNetwork"
        self.num_classes = num_classes
        self.layer_1 = torch.nn.Linear(
            dimensions[1] * dimensions[0] * in_channels,
            128
        )
        self.layer_2 = torch.nn.Linear(128, 256)
        # TODO configure output dims as param
        self.layer_3 = torch.nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(0.25)
        torch.nn.init.xavier_uniform_(self.layer_1.weight)
        torch.nn.init.xavier_uniform_(self.layer_2.weight)
        torch.nn.init.xavier_uniform_(self.layer_3.weight)

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, -1)
        x = self.dropout(x)
        x = self.layer_1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.layer_2(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.layer_3(x)

        return x


