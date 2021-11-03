import numpy as np
from scipy.ndimage.morphology import binary_dilation

from aicsshparam.shtools import align_image_2d
from aicsshparam.shparam import get_shcoeffs
from aicsimageprocessing import cell_rigid_registration


def angle(img, channel_map, /, channel, skew_adjust):
    channel = channel_map[channel]
    return align_image_2d(img, channel, make_unique=skew_adjust,
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
    _center_of_mass = np.mean(
        np.stack(np.where(img[channel] > 0)),
        axis=1
    )
    return np.floor(_center_of_mass + 0.5).astype(int).tolist()


def shcoeffs(img, channel_map, /, channel, lmax=4, sigma=0, compute_lcc=True,
             alignment_2d=False, make_unique=False, prefix=None):
    channel = channel_map[channel]
    (coeffs, _), _ = get_shcoeffs(image=img[channel], lmax=lmax,
                                  sigma=sigma, compute_lcc=compute_lcc,
                                  alignment_2d=alignment_2d, make_unique=make_unique)

    if prefix is not None and len(prefix) > 0:
        coeffs = {f"{prefix}_{k}":v for k,v in coeffs.items()}
    return coeffs
