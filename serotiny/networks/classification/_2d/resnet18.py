"""
ResNet18 module, leveraging pretrained weights +
"""

from collections import OrderedDict

import torch
from torch import nn
import torchvision.models as models


class ResNet18Network(nn.Module):
    """
    A pytorch nn Module that implement a Resnet18 network
    by adding a Conv2D layer to the front of a pretrained
    resnet18 network and a linear layer at the end

    Parameters
    ----------
    in_channels: int
        Number of input channels for the first layer.
        Example: 3

    num_classes: int
        Number of classes for the final layer
        Example: 5

    dimensions: tuple
        Dimensions of input image

    pretrained: bool
        If true, return model pretrained on ImageNet

    """

    def __init__(
        self, in_channels=int, num_classes=int, dimensions=tuple, pretrained=bool
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
