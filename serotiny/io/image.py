from upath import UPath as Path
from typing import Callable, Optional, Sequence, Type, Union

import aicsimageio
import numpy as np
import torch
from aicsimageio.aics_image import _load_reader
from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter
from ome_zarr.reader import Reader
from ome_zarr.io import parse_url


def infer_dims(img: aicsimageio.AICSImage):
    """Given a an AICSImage image, infer the corresponding dimensions."""
    dims = dict(img.dims.items())

    if dims.get("T", 1) > 1:
        raise ValueError("Expected image with no T dimensions")
    if (dims["X"] < 1) or (dims["Y"] < 1):
        raise ValueError("Expected image with height and width")

    if dims.get("S", 0) > 1 and dims.get("C", 0) > 1:
        raise ValueError("Can't handle S dimension > 1 AND C dimension > 1")

    zero_dims = {"T": 0}
    order = "YX"
    for dim in list("ZCS"):
        if dims.get(dim, 1) == 1:
            zero_dims[dim] = 0
        else:
            order = dim + order

    if "C" not in order and "S" not in order:
        order = "C" + order
        del zero_dims["C"]

    return order, zero_dims


def image_loader(
    path,
    select_channels: Optional[Union[Sequence[int], Sequence[str]]] = None,
    dtype: Optional[Union[str, Type[np.number]]] = None,
    transform: Optional[Callable] = None,
    return_channels: bool = False,
    return_as_torch: bool = True,
    reader: Optional[str] = None,
    force_3d: bool = False,
    ome_zarr_level: int = 0,
    ome_zarr_image_name: str = "default",
):
    """Load image from path given by `path`. If the given image doesn't have
    channel names, `select_channels` must be a list of integers.

    Parameters
    ----------
    path: str
        Path of the image to load

    select_channels: Optional[Union[Sequence[int], Sequence[str]]] = None
        Channels to be retrieved from the image. If the given image doesn't
        have channel names, `select_channels` must be a list of integers

    dtype: Type[np.number] = np.float32
        Numpy dtype of output image

    transform: Optional[Callable] = None
        Transform to apply before returning the image

    return_channels: bool = False
        Flag to determine whether to return a channel-index map when loading
        the image. This is only useful when channels have names

    return_as_torch: bool = True
        Flag to determine whether to return the resulting image as a torch.Tensor

    force_3d: bool = True
        Force interpretation as 3d image. Useful for wrongfully stored or
        inferred dimensions.

    ome_zarr_level: int = 0
        If reading an ome zarr image, this specifies what level of the pyramid
        to return

    ome_zarr_image_name: str = "default"
        If reading an ome zarr image, this specifies the image name
    """

    if str(path).endswith(".zarr"):
        img = read_ome_zarr(path, ome_zarr_level, ome_zarr_image_name)
    else:
        if reader is not None:
            reader = _load_reader(reader)
        img = aicsimageio.AICSImage(path, reader=reader)
    channel_names = img.channel_names or list(range(img.data.shape[0]))

    if (select_channels is None) or (len(select_channels) == 0):
        select_channels = channel_names

    if not set(select_channels).issubset(channel_names):
        raise KeyError(
            "Some elements of `select_channels` aren't available: "
            f"\tavailable channels: {channel_names}\n"
            f"\tselect_channels: {select_channels}"
        )

    loaded_channels = select_channels
    loaded_channels_idx = [channel_names.index(channel) for channel in loaded_channels]

    order, zero_dims = infer_dims(img)

    first_dim = {order[0]: loaded_channels_idx} if order[0] in ["S", "C"] else {}

    data = img.get_image_data(order, **zero_dims, **first_dim)

    channel_map = {
        channel_name: index for index, channel_name in enumerate(loaded_channels)
    }

    if len(data.shape) == 3 and force_3d:
        data = np.expand_dims(data, axis=0)

    if dtype is not None:
        data = data.astype(dtype)

    if transform:
        data = transform(data)

    if return_as_torch:
        import torch

        if data.dtype.kind == "u":
            data = data.astype(data.dtype.str[1:])
        data = torch.tensor(data)

    if return_channels:
        return data, channel_map

    return data


def read_ome_zarr(path, level=0, image_name="default"):
    """Function that reads an ome.zarr image and (optionally) returns an
    AICSImage object.

    Note: This is here temporarily, until aicsimageio
    provides an OmeZarrReader class
    """
    path = str(path if image_name is None else Path(path) / image_name)
    reader = Reader(parse_url(path))

    node = next(iter(reader()))
    pps = node.metadata["coordinateTransformations"][0][0]["scale"][-3:]

    return aicsimageio.AICSImage(
        node.data[level].compute(),
        channel_names=node.metadata["name"],
        physical_pixel_sizes=pps,
    )


def tiff_writer(
    img: Union[np.array, torch.Tensor],
    path: Union[str, Path],
    channel_names: Optional[Sequence] = None,
    dim_order: Optional[str] = None,
):
    """Write an image to disk.

    Parameters
    ----------

    img: Union[np.array, torch.Tensor]
        An array/tensor containing the image to be saved

    path: Union[str, Path]
        The path where the image will be saved

    channel_names: Optional[Sequence] = None
        Optional list of labels to assign to each channel
        (in the correct order)
    """

    if dim_order is None:
        if len(img.shape) == 4:
            dim_order = "CZYX"
        elif len(img.shape) == 3:
            dim_order = "CYX"
        else:
            raise ValueError(f"Unexpected image shape {img.shape}")

    if len(dim_order) != len(img.shape):
        raise ValueError(
            f"The dimension spec ({dim_order}) is incompatible "
            f"with the image shape {img.shape}"
        )

    if channel_names is not None:
        channel_names = [channel_names]

    OmeTiffWriter.save(
        data=img, uri=path, channel_names=channel_names, dim_order=dim_order
    )
