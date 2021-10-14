"""
ResNet18 module, leveraging pretrained weights +
"""

from typing import Sequence
from collections import OrderedDict

import torch
from torch import nn
import torchvision.models as models

from ._resnet_utils import resnet18 as make_3D_resnet18


def make_3D_resnet_from_2D(depth):
    """
    Util to instantiate a 3d resnet from a pretrained 2d resnet.
    Replicate the weights of a pretrained 2d resnet across the extra dimension
    in the 3d model
    """
    resnet_3d = make_3D_resnet18(sample_size=224, sample_duration=depth)
    resnet_2d = models.resnet18(pretrained=True)

    # copy resnet2d weights across the 3rd dimension in the 3d model
    kernel_size = resnet_3d.conv1.weight.shape[2]
    for i in range(kernel_size):
        resnet_3d.conv1.weight[:, :, i, :, :].data.copy_(resnet_2d.conv1.weight.data)

    for layer in ["layer1", "layer2", "layer3", "layer4"]:
        for basic_block in ["0", "1"]:
            for conv in ["conv1", "conv2"]:
                weight3d = (
                    resnet_3d._modules[layer]
                    ._modules[basic_block]
                    ._modules[conv]
                    .weight
                )
                weight2d = (
                    resnet_2d._modules[layer]
                    ._modules[basic_block]
                    ._modules[conv]
                    .weight
                )

                kernel_size = weight3d.shape[2]
                for i in range(kernel_size):
                    weight3d[:, :, i, :, :].data.copy_(weight2d.data)

    return resnet_3d


class ResNet18(nn.Module):
    def __init__(
        self,
        in_channels: int,
        input_dims: Sequence[int],
        num_classes: int,
        pretrained=True,
    ):
        """
        Instantiate a 3D Resnet for classification

        Parameters
        ----------
        in_channels: int
            Number of input channels
        input_dims: Sequence[int]
            Dimensions of input channels
        num_classes: int
            Number of classes in the dataset
        pretrained: bool
            Flag to decide whether to train model from scratch or initialize by
            leveraging a pretrained 2d resnet

        """
        super().__init__()
        self.num_classes = num_classes
        self.feature_extractor_first_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv_before_resnet",
                        nn.Conv3d(
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

        # the layer arithmetic for the resnet assumes the inputs are at least
        # 224 wide/high/deep. here, I'm padding the input with zeros so it fulfills that
        # requirement.
        # I'm doing symmetric padding for each dimension, so I calculate how much it
        # lacks until it meets 224 and divide it by 2 (to add half on
        # top/bottom left/right etc etc).
        # the resulting padding tuple has 6 entries because it looks like
        # (pad_x_left, pad_x_right, pad_y_up, pad_y_down, pad_z_top, pad_z_bottom),
        # and I initialize it to all zeros so the cases that don't need padding get
        # bypassed.
        padding = [0] * 6
        for ix, dim in enumerate(input_dims):
            if dim < 224:
                padding[ix] = (224 - dim) // 2
                padding[ix + 1] = (224 - dim) // 2

        self.zero_padding = torch.nn.ConstantPad3d(tuple(padding), 0)

        if pretrained:
            self.classifier = make_3D_resnet_from_2D(input_dims[0])
        else:
            self.classifier = make_3D_resnet18(
                sample_size=224, sample_duration=input_dims[0]
            )

        # Replace final layer
        self.classifier.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.feature_extractor_first_layer(x)
        x = self.zero_padding(x)
        return self.classifier(x)
