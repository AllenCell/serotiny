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
        print(f'  x.shape = {x.shape}')
        x = self.conv1(x)
        print(f'  x.shape (conv1) = {x.shape}')
        x = self.bn1(x)
        print(f'  x.shape (bn1)= {x.shape}')
        x = self.relu1(x)
        print(f'  x.shape (relu1) = {x.shape}')
        x = self.conv2(x)
        print(f'  x.shape (conv2) = {x.shape}')
        x = self.bn2(x)
        print(f'  x.shape (bn2) = {x.shape}')
        x = self.relu2(x)
        print(f'  x.shape (relu2) = {x.shape}')
        return x
