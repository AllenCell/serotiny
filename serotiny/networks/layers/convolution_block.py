from typing import Optional, Union, Dict
from torch import nn
from torch.nn.utils import spectral_norm

from .skip_connection import SkipConnection

def conv_block(
    in_c: int,
    out_c: int,
    kernel_size: int = 3,
    padding: int = 0,
    up_conv: bool = False,
    non_linearity: Optional[nn.Module] = None,
    skip_connection: bool = False,
    batch_norm: bool = True,
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
    batch_norm_cls = nn.Identity
    if mode == "2d":
        conv = nn.ConvTranspose2d if up_conv else nn.Conv2d
        if batch_norm:
            batch_norm_cls = nn.BatchNorm2d
    elif mode == "3d":
        conv = nn.ConvTranspose3d if up_conv else nn.Conv3d
        if batch_norm:
            batch_norm_cls = nn.BatchNorm3d
    else:
        raise ValueError(f"Mode must be '2d' or '3d'. You passed '{mode}'")

    if non_linearity is None:
        non_linearity = nn.ReLU()

    block = nn.Sequential(
        conv(in_c, out_c, kernel_size=kernel_size, padding=padding),
        non_linearity,
        batch_norm_cls(out_c),
    )

    if skip_connection:
        block = SkipConnection(block)
    return block


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_c: int,
        out_c: int,
        kernel_size: int = 3,
        padding: int = 0,
        up_conv: bool = False,
        non_linearity: Optional[nn.Module] = None,
        skip_connection: bool = False,
        batch_norm: bool = False,
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
            batch_norm=batch_norm,
            skip_connection=skip_connection,
            mode=mode,
        )

    def forward(self, x):
        return self.block(x)
