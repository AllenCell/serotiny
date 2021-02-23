"""
ResNet18 module, leveraging pretrained weights +
"""

from collections import OrderedDict

import torch
from torch import nn
import torchvision.models as models


class ResNet18Network(nn.Module):
    def __init__(
        self, in_channels=3, num_classes=5, dimensions=(176, 104), pretrained=True
    ):
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

        self.classifier = models.resnet18(pretrained=pretrained)

        # Replace final layer
        self.classifier.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.feature_extractor_first_layer(x)
        return self.classifier(x)
