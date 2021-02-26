from pathlib import Path
from typing import Union
import numpy as np
import torch

from imageio import imwrite

import aicsimageio
import aicsimageio.transforms as transforms
import aicsimageio.writers.ome_tiff_writer as ome_tiff_writer
import aicsimageprocessing

from skimage.transform import resize

ALL_CHANNELS = ["C", "Y", "X", "S", "T", "Z"]
EMPTY_INDEXES = {channel: 0 for channel in ALL_CHANNELS}

# Set RGB colors
# This will set:
# Membrane to Red
# Structure to Green
# DNA to Blue
DEFAULT_CHANNELS = ["membrane", "structure", "dna"]
CHANNEL_COLORS = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

DEFAULT_TRANSFORM = "CZYX"
TRANSFORM_AXIS_MAP = {"X": "CXZY", "Y": "CYZX", "Z": "CZYX"}


def define_channels(channel_order, indexes=None):
    """
    Take a list of channels and optional indexes and returns
    the necessary arguments to AICSImage.get_image_data()
    """

    if channel_order is None:
        channel_order = []

    orientation = "".join(channel_order) if channel_order else None
    channels = EMPTY_INDEXES.copy()
    if indexes is not None:
        channels.update(indexes)
    for channel in channel_order:
        if isinstance(channels[channel], int):
            del channels[channel]
    return orientation, channels


def normalize_img_zero_one(img_arr):
    """
    min-max-scale each channel in an image.
    """
    for i, channel in enumerate(img_arr.astype(np.float64)):
        img_arr[i] = (channel - channel.min()) / (channel.max() - channel.min())
    return img_arr


def png_loader(
    path_str, channel_order=None, indexes=None, transform=None, output_dtype=np.float64
):
    """
    Load an image from a png file given by `path_str` into a torch tensor.
    ---
    Parameters:
      path_str: str -> path of the image to load
      channel_order: list (optional) -> order in which to load the channels
      indexes: TODO: explain this
      transform: callable (optional) -> transform to apply before returning
      output_dtype: numpy dtype
    Returns:
      torch.Tensor
    """
    # find the orientation and channel indexes
    orientation, channel_indexes = define_channels(channel_order, indexes)

    aicsimageio.use_dask(False)
    img = aicsimageio.AICSImage(path_str)

    # Adding this as a separate line since it may
    # speed up array loading
    img.data

    # use provided orientation to load the image
    if orientation:
        img = img.get_image_data(orientation, **channel_indexes)
    else:
        img = img.get_image_data(**channel_indexes)

    img = img.astype(output_dtype)
    img = normalize_img_zero_one(img)
    img = torch.tensor(img)
    if transform:
        img = transform(img)

    return img

def tiff_loader_CZYX(
    path_str,
    select_channels=None,
    output_dtype=np.float32,
    channel_masks=None,
    mask_thresh=0,
    transform=None,
):
    """
    Load TIFF image from path given by `path_str`.
    ---
    Parameters:
      path_str: str -> path of the image to load
      select_channels: list -> channels to be retrieved from the image
      output_dtype: numpy dtype TODO: explain
      channel_masks: TODO: explain
      mask_thresh: float -> TODO: explain this
      transform: callable (optional) -> transform to apply before returning
    Returns:
      torch.Tensor
    """
    aicsimg = aicsimageio.AICSImage(path_str)
    channel_names = aicsimg.get_channel_names()
    data = aicsimg.get_image_data("CZYX", S=0, T=0)

    if (not set(select_channels).issubset(channel_names)) or (
        not set(channel_masks.keys()).issubset(channel_names)):
        raise KeyError("Some elements of `select_channels` or `channel_masks` "
                       "are not present in `channel_names`:\n"
                       f"\tchannel_names: {channel_names}\n"
                       f"\tchannel_masks: {channel_masks}\n"
                       f"\tselect_channels:: {select_channels}")

    channel_map = {
        channel_name: index for index, channel_name in enumerate(channel_names)
    }

    if channel_masks is not None:
        for channel, mask in channel_masks.items():
            channel_index = channel_map[channel]
            mask_index = channel_map[mask]
            mask = data[mask_index] > mask_thresh
            data[channel_index][~mask] = 0

    if select_channels:
        channel_indexes = [channel_map[channel] for channel in select_channels]
        data = data[channel_indexes, ...]

    data = data.astype(output_dtype)
    if transform:
        data = transform(data)

    return data

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
    data_new = np.zeros((1, num_channels, z_dim_new, y_dim_new, x_dim_new), dtype="uint8")
    for channel in np.arange(num_channels):
        data_new[0, channel, :, :, :] = resize(
            data[channel, :, :, :].squeeze(), (z_dim_new, y_dim_new, x_dim_new),
            preserve_range=True
        )
    data_new = data_new.astype((np.uint8))
    # change this to do it across all channels at once, perhaps this can be done without for loop

    # Write output image
    with ome_tiff_writer.OmeTiffWriter(path_out, overwrite_file=True) as writer2:
        writer2.save(
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
    image_3d = tiff_loader_CZYX(
        path_3d, select_channels=channels or DEFAULT_CHANNELS, channel_masks=masks
    )

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

    # Convert to YXC for PNG writing
    projection = transforms.transpose_to_dims(projection, "CYX", "YXC")

    # Drop size to uint8
    projection = projection.astype(np.uint8)

    # Save to PNG
    imwrite(path_2d, projection)

    return projection.shape
