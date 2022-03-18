from typing import Optional, Union, Dict, Callable
from torch import nn
from torch.nn.utils import spectral_norm

from serotiny.utils import load_config


def conv_block(
    in_c: int,
    out_c: int,
    kernel_size: int = 3,
    padding: int = 0,
    up_conv: bool = False,
    non_linearity=Union[Dict, nn.Module, Callable],
    mode: str = "3d",
):
    """
    Util function to instantiate a convolutional block.

    Parameters
    ----------
    in_c: int
        number of input channels
    out_c: int
        number of output channels
    kernel_size: Sequence[int] (defaults to (3, 3, 3))
        dimensions of the convolutional kernel to be applied
    padding:
        padding value for the convolution (defaults to 0)
    mode:
        Dimensionality of the input data. Can be "2d" or "3d".
    """
    if mode == "2d":
        conv = nn.ConvTranspose2d if up_conv else nn.Conv2d
        batch_norm = nn.BatchNorm2d
    elif mode == "3d":
        conv = nn.ConvTranspose3d if up_conv else nn.Conv3d
        batch_norm = nn.BatchNorm3d
    else:
        raise ValueError(f"Mode must be '2d' or '3d'. You passed '{mode}'")

    if non_linearity is None:
        non_linearity = nn.ReLU()
    elif hasattr(non_linearity, "items"):
        non_linearity = load_config(non_linearity)
    else:
        raise TypeError(f"Unexpected type for `non_linearity`: {type(non_linearity)}")

    return nn.Sequential(
        conv(in_c, out_c, kernel_size=kernel_size, padding=padding),
        non_linearity,
        batch_norm(out_c),
    )


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_c: int,
        out_c: int,
        kernel_size: int = 3,
        padding: int = 0,
        up_conv: bool = False,
        non_linearity=Union[Dict, nn.Module, Callable],
        mode: str = "3d",
    ):
        """
        Convolutional block class.

        Parameters
        ----------
        in_c: int
            number of input channels
        out_c: int
            number of output channels
        kernel_size: Sequence[int] (defaults to (3, 3, 3))
            dimensions of the convolutional kernel to be applied
        padding:
            padding value for the convolution (defaults to 0)
        mode:
            Dimensionality of the input data. Can be "2d" or "3d".
        """

        super().__init__()
        self.block = conv_block(
            in_c=in_c,
            out_c=out_c,
            kernel_size=kernel_size,
            padding=padding,
            up_conv=up_conv,
            non_linearity=non_linearity,
            mode=mode,
        )

    def forward(self, x):
        return self.block(x)
