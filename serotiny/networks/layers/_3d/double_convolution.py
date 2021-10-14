from torch import nn


class DoubleConvolution(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        kernel_size: int = 3,  # Original paper = 3
        stride: int = 1,  # Original paper = no mention, label-free = default
        padding: int = 1,  # Original paper = 0, label-free = 1
    ):
        super().__init__()

        self.conv1 = nn.Conv3d(
            n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.bn1 = nn.BatchNorm3d(n_out)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv3d(
            n_out, n_out, kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.bn2 = nn.BatchNorm3d(n_out)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x
