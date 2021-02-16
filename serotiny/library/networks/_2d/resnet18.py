"""
ResNet18 module, leveraging pretrained weights +
"""

from collections import OrderedDict

import torch
from torch import nn
import torchvision.models as models


class ResNet18Network(nn.Module):
    def __init__(self, in_channels=3, num_classes=5, dimensions=(176, 104)):
        super().__init__()
        self.network_name = "Resnet18"
        self.num_classes = num_classes
        self.feature_extractor_first_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv_before_resnet",
                        nn.Conv2d(
                            in_channels,
                            3,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=False,
                        ),
                    )
                ]
            )
        )
        # initialize weights of first layer
        torch.nn.init.xavier_uniform_(
            self.feature_extractor_first_layer.conv_before_resnet.weight
        )

        self.feature_extractor = models.resnet18(pretrained=True)
        # dont freeze the weights
        # self.feature_extractor.eval()

        # Classifier architecture to put on top of resnet18
        self.classifier = nn.Sequential(
            OrderedDict(
                [
                    (
                        "fc1",
                        nn.Linear(self.feature_extractor.fc.out_features, 100)
                    ),
                    ("relu", nn.ReLU()),
                    # TODO configure output number classes
                    ("fc2", nn.Linear(100, num_classes)),
                    # ("output", nn.Sigmoid()),
                ]
            )
        )

    def forward(self, x):
        x = self.feature_extractor_first_layer(x)
        representations = self.feature_extractor(x)
        x = self.classifier(representations)
        return x

