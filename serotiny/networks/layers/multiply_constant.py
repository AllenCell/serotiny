import torch.nn as nn
import torch
class MultiplyConstant(nn.Module):
    def __init__(self, constant):
        super().__init__()
        self.constant = constant

    def forward(self, x):
        return self.constant * x