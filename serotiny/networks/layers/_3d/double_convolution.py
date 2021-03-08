import torch

class DoubleConvolution(torch.nn.Module):
    def __init__(
            self,
            n_in: int,
            n_out: int,
            kernel_size: int,
            padding: int,
            ):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(n_in, n_out, kernel_size=kernel_size, padding=padding)
        self.bn1 = torch.nn.BatchNorm3d(n_out)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv3d(n_out, n_out, kernel_size=kernel_size, padding=padding)
        self.bn2 = torch.nn.BatchNorm3d(n_out)
        self.relu2 = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

