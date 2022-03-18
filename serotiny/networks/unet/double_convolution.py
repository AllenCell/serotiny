from torch import nn


class DoubleConvolution(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        kernel_size: int = 3,  # Original paper = 3
        stride: int = 1,  # Original paper = no mention, label-free = default
        padding: int = 1,  # Original paper = 0, label-free = 1
        bn=True
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv3d(
                n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding
            ),
            (nn.BatchNorm3d(n_out) if bn else nn.Identity()),
            nn.ReLU(),
            nn.Conv3d(
                n_out, n_out, kernel_size=kernel_size, stride=stride, padding=padding
            ),
            (nn.BatchNorm3d(n_out) if bn else nn.Identity()),
            nn.ReLU(),
        )
        
    def forward(self, x):
        return self.net(x)
