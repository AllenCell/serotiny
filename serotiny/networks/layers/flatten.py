import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x, batch=True):
        if batch:
            return x.reshape(x.shape[0], -1)
        else:
            return torch.flatten(x)
