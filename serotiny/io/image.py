from pathlib import Path
from typing import Union, Sequence, Optional
import numpy as np
import torch

import aicsimageio
import aicsimageio.transforms as transforms
from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter

import aicsimageprocessing

ALL_CHANNELS = ["C", "Y", "X", "S", "T", "Z"]
EMPTY_INDEXES = {channel: 0 for channel in ALL_CHANNELS}

# Set RGB colors
# This will set:
# Membrane to Red
# Structure to Green
# DNA to Blue
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


def png_loader(
    path, channel_order=None, indexes=None, transform=None, output_dtype=np.float64
):
    """
    Load an image from a png file given by `path` into a torch tensor.
    ---
    Parameters:
      path: str -> path of the image to load
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
    img = aicsimageio.AICSImage(path)

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


def infer_dims(img):
    dims = dict(img.dims.items())

    if "S" in dims:
        if dims["S"] > 1:
            raise ValueError("Expected image with no S dimensions")
    if "T" in dims:
        if dims["T"] > 1:
            raise ValueError("Expected image with no T dimensions")

    if (dims["X"] < 1) or (dims["Y"] < 1):
        raise ValueError("Expected image with height and width")

    if dims["Z"] <= 1:
        # 2D tiff
        return "CYX"
    # 3D tiff
    return "CZYX"


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
    channel_names = aicsimg.channel_names
    data = aicsimg.get_image_data("CZYX", S=0, T=0)

    mask_keys = channel_masks.keys() if channel_masks else {}

    if select_channels is None:
        select_channels = channel_names

    if (not set(select_channels).issubset(channel_names)) or (
        not set(mask_keys).issubset(channel_names)
    ):
        raise KeyError(
            "Some elements of `select_channels` or `channel_masks` "
            "are not present in `channel_names`:\n"
            f"\tchannel_names: {channel_names}\n"
            f"\tchannel_masks: {channel_masks}\n"
            f"\tselect_channels:: {select_channels}"
        )

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


def tiff_loader(
    path,
    select_channels=None,
    output_dtype=np.float32,
    channel_masks=None,
    mask_thresh=0,
    transform=None,
    return_channels=False,
):
    """
    Load TIFF image from path given by `path`.

    Parameters
    ----------
    path: str
        path of the image to load
    select_channels: list
        channels to be retrieved from the image
    output_dtype: np.dtype
        numpy dtype of output image
    channel_masks: dict
        dictionary with channel names as keys, and the names of the channels
        that should be used to masked them as values.
    mask_thresh: float
        value under which a pixel value in the mask will signify masking off
        the corresponding pixel in the original channel
    transform: callable (optional)
        transform to apply before returning the image
    return_channels: bool
        flag to determine whether to return a channel-index map when loading
        the image

    """
    aicsimg = aicsimageio.AICSImage(path)
    channel_names = aicsimg.channel_names

    if (select_channels is None) or (len(select_channels) == 0):
        select_channels = channel_names
    if channel_masks is None:
        channel_masks = {}
    mask_keys = channel_masks.keys()

    if (not set(select_channels).issubset(channel_names)) or (
        not set(mask_keys).issubset(channel_names)
    ):
        raise KeyError(
            "Some elements of `select_channels` or `channel_masks` "
            "are not present in `channel_names`:\n"
            f"\tchannel_names: {channel_names}\n"
            f"\tchannel_masks: {channel_masks}\n"
            f"\tselect_channels:: {select_channels}"
        )

    loaded_channels = select_channels + list(channel_masks.values())
    loaded_channels_idx = [channel_names.index(channel) for channel in loaded_channels]

    dims = infer_dims(aicsimg)

    if "Z" in dims:
        data = aicsimg.get_image_data(dims, S=0, T=0, channels=loaded_channels_idx)
    else:
        data = aicsimg.get_image_data(dims, Z=0, S=0, T=0, channels=loaded_channels_idx)

    channel_map = {
        channel_name: index for index, channel_name in enumerate(loaded_channels)
    }

    if channel_masks is not None:
        for channel, mask in channel_masks.items():
            channel_index = channel_map[channel]
            mask_index = channel_map[mask]
            mask = data[mask_index] > mask_thresh
            data[channel_index][~mask] = 0

    data = data.astype(output_dtype)
    if transform:
        data = transform(data)

    if return_channels:
        return data, channel_map
    return data


def tiff_writer(
    img,
    path: Union[str, Path],
    channels: Optional[Sequence] = None,
    overwrite: bool = False,
):

    if len(img.shape) == 4:
        dims = "CZYX"
    elif len(img.shape) == 3:
        dims = "CYX"
    else:
        raise ValueError(f"Unexpected image shape {img.shape}")

    # Write output image
    OmeTiffWriter.save(
        data=img, uri=path, channel_names=channels, dimension_order="STCZYX"
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
    channel_names = aicsimg.channel_names
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
    OmeTiffWriter.save(
        data=data_new,
        uri=path_out,
        channel_names=channel_names,
        dimension_order="STCZYX",
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


def subset_channels(
    channel_subset: Sequence[Union[int, str]], channels: Sequence[Union[int, str]]
):
    """
    Subset channels given a list of both

    Parameters
    -----------
    channel_subset: Sequence[Union[int, str]]
        List of subset channels

    channels: Sequence[Union[int, str]]
        List of all channels

    Returns:
    channel_indexes:
        Indexes of subset channels in original channel list

    num_channels:
        New length of channels
    """
    if channel_subset is not None:
        try:
            channel_indexes = [
                channels.index(channel_name) for channel_name in channel_subset
            ]
            num_channels = len(channel_indexes)
        except ValueError:
            raise Exception(
                (
                    f"channel indexes {channel_subset} "
                    f"do not match channel names {channels}"
                )
            )
    return channel_indexes, num_channels
