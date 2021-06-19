from typing import Dict

import torch.nn as nn
from serotiny.utils import get_classes_from_config

class Sequential(nn.Module):
    """
    Auxiliary class to allow instantiation of a composition of networks from
    a yaml config
    """
    def __init__(
        self,
        modules: Dict,
    ):
        super().__init__()
        modules = get_classes_from_config(modules)
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)
