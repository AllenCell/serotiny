from typing import Dict

import torch.nn as nn
from serotiny.utils import path_invocations

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
        modules = path_invocations(modules)
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)
