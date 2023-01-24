import torch.nn as nn


class SkipConnection(nn.Module):
    def __init__(self, wrapped_module):
        super().__init__()
        self.wrapped_module = wrapped_module

    def forward(self, x):
        return x + self.wrapped_module(x)
