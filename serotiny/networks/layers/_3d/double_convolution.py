import torch
from torch import nn

# In original paper, kernel_size = 3 and padding = 0
class DoubleConvolution(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        
        kernel_size: int = 3,  # Original paper = 3
        stride: int = 1,       # Original paper = no mention, label-free = default
        padding: int = 1,      # Original paper = 0, label-free = 1
    ):
        super().__init__()

        self.conv1 = nn.Conv3d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm3d(n_out)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv3d(n_out, n_out, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm3d(n_out)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        print(f"  x.shape = {x.shape}")
        
        x = self.conv1(x)
        print(f"  x.shape (conv1) = {x.shape}")
        
        x = self.bn1(x)
        print(f"  x.shape (bn1)= {x.shape}")
        
        x = self.relu1(x)
        print(f"  x.shape (relu1) = {x.shape}")
        
        x = self.conv2(x)
        print(f"  x.shape (conv2) = {x.shape}")
        
        x = self.bn2(x)
        print(f"  x.shape (bn2) = {x.shape}")
        
        x = self.relu2(x)
        print(f"  x.shape (relu2) = {x.shape}")
        
        return x
