from typing import Union, Sequence, Optional, Type, Callable
from pathlib import Path

import torch
import numpy as np

import aicsimageio
from aicsimageio.aics_image import _load_reader
from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter


def infer_dims(img: aicsimageio.AICSImage):
    """
    Given a an AICSImage image, infer the corresponding
    tiff dimensions.
    """
    dims = dict(img.dims.items())

    if "S" in dims:
        if dims["S"] > 1:
            raise ValueError("Expected image with no S dimensions")
    if "T" in dims:
        if dims["T"] > 1:
            raise ValueError("Expected image with no T dimensions")

    if (dims["X"] < 1) or (dims["Y"] < 1):
        raise ValueError("Expected image with height and width")

    return "CYX" if dims["Z"] <= 1 else "CZYX"


def image_loader(
    path,
    select_channels: Optional[Union[Sequence[int], Sequence[str]]] = None,
    output_dtype: Optional[Type[np.number]] = None,
    transform: Optional[Callable] = None,
    return_channels: bool = False,
    reader: Optional[str] = None,
):
    """
    Load image from path given by `path`. If the given image doesn't have channel
    names, `select_channels` must be a list of integers

    Parameters
    ----------
    path: str
        Path of the image to load

    select_channels: Optional[Union[Sequence[int], Sequence[str]]] = None
        Channels to be retrieved from the image. If the given image doesn't
        have channel names, `select_channels` must be a list of integers

    output_dtype: Type[np.number] = np.float32
        Numpy dtype of output image

    transform: Optional[Callable] = None
        Transform to apply before returning the image

    return_channels: bool = False
        Flag to determine whether to return a channel-index map when loading
        the image. This is only useful when channels have names
    """

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

    dims = infer_dims(img)

    if "Z" in dims:
        data = img.get_image_data(dims, S=0, T=0, channels=loaded_channels_idx)
    else:
        data = img.get_image_data(dims, Z=0, S=0, T=0, channels=loaded_channels_idx)

    channel_map = {
        channel_name: index for index, channel_name in enumerate(loaded_channels)
    }

    if output_dtype is not None:
        data = data.astype(output_dtype)

    if transform:
        data = transform(data)

    if return_channels:
        return data, channel_map

    return data


def tiff_writer(
    img: Union[np.array, torch.Tensor],
    path: Union[str, Path],
    channel_names: Optional[Sequence] = None,
    dim_order: Optional[str] = None,
):
    """
    Write an image to disk.

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
