from typing import Union, Sequence, Optional, Type, Callable
from pathlib import Path

import torch
import numpy as np

import aicsimageio
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

    img = aicsimageio.AICSImage(path)
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


def change_resolution(
    path_in: Union[str, Path],
    path_out: Union[str, Path],
    ZYX_resolution: Union[float, list],
):
    """
    Changes the resolution of a 3D OME TIFF file
    Parameters
    ----------
    path_in: Union[str, Path]
        The path to the input OME TIFF file
    path_out: Union[str, Path]
        The path to the output OME TIFF file
    ZYX_resolution: Union[float, list]
        Resolution scaling factor or desired ZYX dimensions (list of 3)
    Returns
    -------
    data_new.shape: Tuple
        Tuple that contains the image dimensions of output image
    """

    aicsimageio.use_dask(False)

    # Read in image and get channel names
    aicsimg = aicsimageio.AICSImage(path_in)
    channel_names = aicsimg.get_channel_names()
    data = aicsimg.get_image_data("CZYX", S=0, T=0)
    # this function should change when we have multiple scenes (S>0) or time series (T>0)

    # Get image dimensions
    num_channels, z_dim, y_dim, x_dim = data.shape

    # Get image dimensions of new image
    if isinstance(ZYX_resolution, list):
        if len(ZYX_resolution) != 3:
            raise ValueError(
                f"Resolution must be three long (Z Y X) not {len(ZYX_resolution)}"
            )
        z_dim_new, y_dim_new, x_dim_new = ZYX_resolution
    else:
        z_dim_new = np.round(z_dim * ZYX_resolution).astype(np.int)
        y_dim_new = np.round(y_dim * ZYX_resolution).astype(np.int)
        x_dim_new = np.round(x_dim * ZYX_resolution).astype(np.int)
    # Resize to get desired resolution
    data_new = np.zeros(
        (1, num_channels, z_dim_new, y_dim_new, x_dim_new), dtype="uint8"
    )
    for channel in np.arange(num_channels):
        data_new[0, channel, :, :, :] = resize(
            data[channel, :, :, :].squeeze(),
            (z_dim_new, y_dim_new, x_dim_new),
            preserve_range=True,
        )
    data_new = data_new.astype((np.uint8))
    # change this to do it across all channels at once, perhaps this can be done without for loop

    # Write output image
    with OmeTiffWriter(path_out, overwrite_file=True) as writer:
        writer.save(
            data=data_new, channel_names=channel_names, dimension_order="STCZYX"
        )

    return data_new.shape


def project_2d(
    path_3d,
    axis,
    method,
    path_2d,
    channels=None,
    masks=None,
    proj_all=False,
):
    """
    Apply 2d projection to 3d image in path given by `path_3d`
    Parameters
    ----------
    path_3d: Union[str, Path]
        The path to the input OME TIFF file
    axis: str
        The axis across which to project
    method: str
        The method for the projection
    path_2d: Union[str, Path]
        The path in which to save the projection image
    channels:
        TODO explain
    masks:
        TODO explain
    proj_all: bool
        TODO explain
    Returns
    -------
    projection.shape: Tuple
        Tuple that contains the image dimensions of output projection
    """
    aicsimageio.use_dask(False)

    # load the 3d image
    image_3d = tiff_loader(path_3d, select_channels=channels, channel_masks=masks)

    # find the argument based on the axis
    if axis in TRANSFORM_AXIS_MAP:
        axis_transform = TRANSFORM_AXIS_MAP[axis]
    else:
        raise Exception(
            f"only axes available are {list(TRANSFORM_AXIS_MAP.keys())}, not {axis}"
        )

    # choose another transform if we aren't doing the Z axis
    if axis_transform != DEFAULT_TRANSFORM:
        image_3d = transforms.transpose_to_dims(
            image_3d, DEFAULT_TRANSFORM, axis_transform
        )

    # project from CZYX to CYX
    projection = aicsimageprocessing.imgtoprojection(
        image_3d,
        proj_all=proj_all,
        proj_method=method,
        local_adjust=False,
        global_adjust=True,
        colors=CHANNEL_COLORS,
    )

    # Drop size to uint8
    projection = projection.astype(np.uint8)

    # Save to TIFF
    tiff_writer(projection, path_2d, channels)

    return projection.shape
