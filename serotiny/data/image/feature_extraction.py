import numpy as np
from scipy.ndimage.morphology import binary_dilation

from aicsshparam.shtools import align_image_2d
from aicsimageprocessing import cell_rigid_registration

#from cvapipe_analysis.steps.compute_features.compute_features_tools import (
#    FeatureCalculator
#)


def _get_channel_ixs(channel_map, /, channels):
    if len(channels) == 0:
        return channel_map[channels[0]]
    else:
        return [channel_map[ch] for ch in channels]


def angle(img, channel_map, /, channels, skew_adjust):
    channel = _get_channel_ixs(channel_map, channels)
    return align_image_2d(img, 0, make_unique=skew_adjust,
                          compute_aligned_image=False)


def min_max(img, channel_map, /, channels):
    results = {}
    for channel in channels:
        channel_ix = channel_map[channel]
        results[f"{channel}_min"] = img[channel_ix].min()
        results[f"{channel}_max"] = img[channel_ix].max()

    return results


def percentile(img, channel_map, /, channels, up_p, low_p):
    channels = range(img.shape[0])
    results = {}
    for channel in channels:
        channel_ix = channel_map[channel]
        results[f"{channel}_0.5_perc"] = np.percentile(img[channel_ix], up_p)
        results[f"{channel}_99.5_perc"] = np.percentile(img[channel_ix], low_p)

    return results


def bbox(img, channel_map, /, channels):
    img = img[channels]
    channel_com, channel_crop = channels

    channel_com = channel_map[channel_com]
    channel_crop = channel_map[channel_crop]

    img, _, _ = cell_rigid_registration(
        img, ch_crop=channel_crop, ch_angle=None, ch_com=channel_com,
        align_image=False, ch_flipdim=None
    )

    return img.shape.tolist()


def dillated_bbox(img, channel_map, /, channel, structuring_element=[5,5,5]):
    channel_ix = channel_map[channel]
    img = img[channel_ix]
    img = binary_dilation(img, np.ones(structuring_element))

    non_zero = np.argwhere(img)
    return (non_zero.max(axis=0) - non_zero.min(axis=0)).tolist()


def center_of_mass(img, channel_map, /, channel):
    channel = channel_map[channel]
    center_of_mass = np.mean(
        np.stack(np.where(img[channel] > 0)),
        axis=1
    )
    return np.floor(center_of_mass + 0.5).astype(int).tolist()


def spharm(img, channel_map, /, channels, channel, cvapipe_analysis_config):
    channel = _get_channel_ixs(channel_map, channels)
    config = cvapipe_analysis_config

    # add dummy parameter. needed to be able to instantiate FeatureCalculator
    config["project"] = dict(local_staging="/dev/null")
    fc = FeatureCalculator(config)

    channel = channel_map[channel]

    return fc.get_features_from_binary_image(
        input_image=img[channel],
        input_reference_image=None,
        compute_shcoeffs=True
    )
