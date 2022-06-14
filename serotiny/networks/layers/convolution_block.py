from typing import Optional

from torch import nn


def conv_block(
    in_c: int,
    out_c: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 0,
    up_conv: bool = False,
    non_linearity: Optional[nn.Module] = None,
    batch_norm: bool = True,
    mode: str = "3d",
):
    """Util function to instantiate a convolutional block.

    Parameters
    ----------
    in_c: int
        number of input channels
    out_c: int
        number of output channels
    kernel_size: Sequence[int] (defaults to (3, 3, 3))
        dimensions of the convolutional kernel to be applied
    stride: Sequence[int] (defaults to (1, 1, 1))
        stride of the convolution
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
        conv(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding),
        non_linearity,
        batch_norm_cls(out_c),
    )

    return block


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_c: int,
        out_c: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        up_conv: bool = False,
        non_linearity: Optional[nn.Module] = None,
        skip_connection: bool = False,
        batch_norm: bool = False,
        mode: str = "3d",
    ):
        """Convolutional block class.

        Parameters
        ----------
        in_c: int
            number of input channels
        out_c: int
            number of output channels
        kernel_size: Sequence[int] (defaults to (3, 3, 3))
            dimensions of the convolutional kernel to be applied
        stride: Sequence[int] (defaults to (1, 1, 1))
            stride of the convolution
        padding:
            padding value for the convolution (defaults to 0)
        mode:
            Dimensionality of the input data. Can be "2d" or "3d".
        """

        super().__init__()
        self.skip_connection = skip_connection
        self.block = conv_block(
            in_c=in_c,
            out_c=out_c,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            up_conv=up_conv,
            non_linearity=non_linearity,
            batch_norm=batch_norm,
            mode=mode,
        )

    def forward(self, x):
        res = self.block(x)
        if self.skip_connection and (res.shape[1] == x.shape[1]):
            return res + nn.functional.interpolate(x, res.shape[2:])
        else:
            return res
