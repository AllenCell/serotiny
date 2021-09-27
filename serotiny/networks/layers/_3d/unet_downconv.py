from torch import nn

from .double_convolution import DoubleConvolution


class DownConvolution(nn.Module):
    def __init__(
        self,
        current_depth: int,
        n_in: int,
        n_out: int,
        kernel_size_doubleconv: int = 3,  # Original paper = 3
        stride_doubleconv: int = 1,  # Original paper = no mention, label-free = default
        padding_doubleconv: int = 1,  # Original paper = 0, label-free = 1
        pooling: str = "mean",  # Original paper = max
        kernel_size_pooling: int = 2,  # Original paper = 2
        stride_pooling: int = 2,  # Original paper = no mention, label-free = 2
        padding_pooling: int = 0,  # Original paper = no mention, label-free = default
    ):
        super().__init__()

        self.current_depth = current_depth

        self.double_conv = DoubleConvolution(
            n_in,
            n_out,
            kernel_size=kernel_size_doubleconv,
            stride=stride_doubleconv,
            padding=padding_doubleconv,
        )

        # If we're at the bottom, only instantiate one double_conv and nothing else
        if self.current_depth == 0:
            pass
        else:
            # In original paper, pooling = max, kernel_size = 2 and stride = 2
            if pooling == "mean":
                self.pooling = nn.AvgPool3d(
                    kernel_size=kernel_size_pooling,
                    stride=stride_pooling,
                    padding=padding_pooling,
                )
            elif pooling == "max":
                self.pooling = nn.MaxPool3d(
                    kernel_size=kernel_size_pooling,
                    stride=stride_pooling,
                    padding=padding_pooling,
                )
            else:
                # Use convolution to perform pooling
                self.pooling = nn.Conv3d(
                    n_out,
                    n_out,
                    kernel_size=kernel_size_pooling,
                    stride=stride_pooling,
                    padding=padding_pooling,
                )

            self.batch_norm = nn.BatchNorm3d(n_out)
            self.activation = nn.ReLU()

    def forward(self, x):

        # print(f' x.shape = {x.shape}')

        if self.current_depth != 0:
            x_doubleconv_down = self.double_conv(x)
            # print(f' x_doubleconv_down = {x_doubleconv_down.shape}')

            x_pool_down = self.pooling(x_doubleconv_down)
            # print(f' x_pool_down = {x_pool_down.shape}')

            x_batchnorm_down = self.batch_norm(x_pool_down)
            # print(f' x_batchnorm_down = {x_batchnorm_down.shape}')

            x_activation_down = self.activation(x_batchnorm_down)
            # print(f' x_activation_down = {x_activation_down.shape}')

            x_return = x_activation_down

        else:
            x_doubleconv_down = self.double_conv(x)
            # print(f' x_doubleconv_down = {x_doubleconv_down.shape}')

            x_return = x_doubleconv_down

        return x_return, x_doubleconv_down
