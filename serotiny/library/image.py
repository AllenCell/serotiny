import numpy as np
import json
import torch

from imageio import imwrite

import aicsimageio
import aicsimageio.transforms as transforms
import aicsimageprocessing


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
    for i, c in enumerate(img_arr.astype(np.float64)):
        img_arr[i] = c / c.max()
    return img_arr


def png_loader(
    path_str,
    channel_order=None,
    indexes=None,
    transform=None,
    output_dtype=np.float64
):
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
    channel_indexes=None,
    select_channels=None,
    output_dtype=np.float32,
    channel_masks=None,
    mask_thresh=0,
    transform=None,
):
    aicsimg = aicsimageio.AICSImage(path_str)
    channel_names = aicsimg.get_channel_names()
    data = aicsimg.get_image_data("CZYX", S=0, T=0)

    channel_map = {
        channel_name: index for index, channel_name in enumerate(channel_names)
    }

    def map_channel(channel):
        if channel in channel_map:
            return channel_map[channel]
        else:
            raise Exception(
                (
                    f"the supplied channel - {channel},"
                    f" does not match existing channels: {channel_names}"
                )
            )

    if channel_masks is not None:
        for channel, mask in channel_masks.items():
            channel_index = map_channel(channel)
            mask_index = map_channel(mask)
            mask = data[mask_index] > mask_thresh
            data[channel_index][~mask] = 0

    if select_channels:
        channel_indexes = [map_channel(channel) for channel in select_channels]

    if channel_indexes:
        data = data[channel_indexes, ...]

    data = data.astype(output_dtype)
    if transform:
        data = transform(data)

    return data


def project_2d(path_3d, axis, method, path_2d, channels=None, masks=None):
    aicsimageio.use_dask(False)

    # load the 3d image
    image_3d = tiff_loader_CZYX(
        path_3d,
        select_channels=channels or DEFAULT_CHANNELS,
        channel_masks=masks
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
        proj_all=False,
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


def tiff_loader_CYX(
    path_str,
    channel_indexes=[4],
    center_slab_width=8,
    output_dtype=np.float32,
    channel_mask_inds={4: 1},
    mask_thresh=0,
):
    data = tiff_loader_CZYX(
        path_str,
        channel_indexes=channel_indexes,
        output_dtype=output_dtype,
        channel_mask_inds=channel_mask_inds,
        mask_thresh=mask_thresh,
    )
    data = normalize_img_zero_one(data)
    data = data[
        :,
        (data.shape[1] - center_slab_width)
        // 2: (data.shape[1] + center_slab_width)
        // 2,
        ...,
    ]
    data = data.max(axis=1)
    data = data.astype(output_dtype)
    return data


def tiff_loader_CZYX_seg(
    path_str,
    channel_indexes=[0, 1],
    seg_thresh=0,
    output_dtype=np.float32,
    channel_mask_inds={},
    mask_thresh=0,
):
    data = tiff_loader_CZYX(
        path_str,
        channel_indexes=channel_indexes,
        output_dtype=output_dtype,
        channel_mask_inds=channel_mask_inds,
        mask_thresh=mask_thresh,
    )
    data = data > seg_thresh
    data = data.astype(output_dtype)
    return data


def tiff_loader_CYX_seg(
    path_str,
    channel_indexes=[0, 1],
    seg_thresh=0,
    center_slab_width=8,
    output_dtype=np.float32,
    channel_mask_inds={},
    mask_thresh=0,
):
    data = tiff_loader_CZYX(
        path_str,
        channel_indexes=channel_indexes,
        output_dtype=output_dtype,
        channel_mask_inds=channel_mask_inds,
        mask_thresh=mask_thresh,
    )
    data = data[
        :,
        (data.shape[1] - center_slab_width)
        // 2: (data.shape[1] + center_slab_width)
        // 2,
        ...,
    ]
    data = data > seg_thresh
    data = data.max(axis=1)
    data = data.astype(output_dtype)
    return data


def feature_loader(
    path_str,
    feature_dtype_dict={"dna_volume": np.float32, "cell_volume": np.float32}
):
    with open(path_str) as json_file:
        data = json.load(json_file)
    return {k: v(data[k]) for k, v in feature_dtype_dict.items()}
