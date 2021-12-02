from typing import Union, Dict, Sequence, Optional
import torch
import numpy as np
from scipy.ndimage.morphology import binary_dilation

from aicsshparam.shtools import align_image_2d
from aicsshparam.shparam import get_shcoeffs
from aicsimageprocessing import cell_rigid_registration

ArrayLike = Union[np.array, torch.Tensor]

def angle(
    img: ArrayLike,
    channel_map: Dict,
    /,
    channel: str,
    skew_adjust: bool
):
    """
    Extract angle of rotation in the X place, for alignment
    of major axes of the max projection of `channel`.

    Parameters
    ----------
    img: ArrayLike
        The image from which to extract the feature

    channel_map: Dict[str, int]
        A dictionary containing channel names and correspoding indices

    channel: str
        The channel name from which to extract the angle

    skew_adjust: bool
        Whether to adjust for skewness
    """
    channel = channel_map[channel]
    return align_image_2d(img, channel, make_unique=skew_adjust,
                          compute_aligned_image=False)


def min_max(
    img: ArrayLike,
    channel_map: Dict[str, int],
    /,
    channels: Sequence[str]
):
    """
    Extract the minimum and maximum values for the given list
    of channels

    Parameters
    ----------
    img: ArrayLike
        The image from which to extract the feature

    channel_map: Dict[str, int]
        A dictionary containing channel names and correspoding indices

    channels: Sequence[str]
        List of channels for which to extract mins and maxes

    """
    results = {}
    for channel in channels:
        channel_ix = channel_map[channel]
        results[f"{channel}_min"] = img[channel_ix].min()
        results[f"{channel}_max"] = img[channel_ix].max()

    return results


def percentile(
    img: ArrayLike,
    channel_map: Dict[str, int],
    /,
    channels: Sequence[str],
    perc: float
):
    """
    Extract a given percentile value for the given list of channels

    Parameters
    ----------
    img: ArrayLike
        The image from which to extract the feature

    channel_map: Dict[str, int]
        A dictionary containing channel names and correspoding indices

    channels: Sequence[str]
        List of channels for which to extract the percentile value

    perc: float
        The percentile to extract

    """
    channels = range(img.shape[0])
    results = {}
    for channel in channels:
        channel_ix = channel_map[channel]
        results[f"{channel}_{perc:.2f}_perc"] = np.percentile(img[channel_ix],
                                                              perc)

    return results


def bbox(
    img: ArrayLike,
    channel_map: Dict[str, int],
    /,
    channels: Sequence[str],
):
    """
    Extract the bounding box, using a pair of channels given
    in `channels`, as [center of mass channel, cropping channel]

    Parameters
    ----------
    img: ArrayLike
        The image from which to extract the feature

    channel_map: Dict[str, int]
        A dictionary containing channel names and correspoding indices

    channels: Sequence[str]
        List of channels for which to extract the bounding box

    """
    img = img[channels]
    channel_com, channel_crop = channels

    channel_com = channel_map[channel_com]
    channel_crop = channel_map[channel_crop]

    img, _, _ = cell_rigid_registration(
        img, ch_crop=channel_crop, ch_angle=None, ch_com=channel_com,
        align_image=False, ch_flipdim=None
    )

    return img.shape.tolist()


def dillated_bbox(
    img: ArrayLike,
    channel_map: Dict[str, int],
    /,
    channel: str,
    structuring_element: Sequence[int] = [5,5,5],
):
    """
    Extract the bounding box of the result of dilating the channel
    given by channel, using the given structuring element

    Parameters
    ----------
    img: ArrayLike
        The image from which to extract the feature

    channel_map: Dict[str, int]
        A dictionary containing channel names and correspoding indices

    channel: str
        The channel name from which to extract the bounding box

    structuring_element: Sequence[int]
        The dillation structuring element shape

    """
    channel_ix = channel_map[channel]
    img = img[channel_ix]
    img = binary_dilation(img, np.ones(structuring_element))

    non_zero = np.argwhere(img)
    return (non_zero.max(axis=0) - non_zero.min(axis=0)).tolist()


def center_of_mass(
    img: ArrayLike,
    channel_map: Dict[str, int],
    /,
    channel: str,
):
    """
    Extract the center of mass of the given channel

    Parameters
    ----------
    img: ArrayLike
        The image from which to extract the feature

    channel_map: Dict[str, int]
        A dictionary containing channel names and correspoding indices

    channel: str
        The channel name from which to extract the center of mass

    """
    channel = channel_map[channel]
    _center_of_mass = np.mean(
        np.stack(np.where(img[channel] > 0)),
        axis=1
    )
    return np.floor(_center_of_mass + 0.5).astype(int).tolist()


def shcoeffs(
    img: ArrayLike,
    channel_map: Dict[str, int],
    /,
    channel: str,
    lmax: int = 4,
    sigma: float = 0,
    compute_lcc: bool = True,
    alignment_2d: bool = False,
    make_unique: bool = False,
    prefix: Optional[str] = None
):
    """
    Compute spherical harmonic coefficients for the given channel

    Parameters
    ----------
    img: ArrayLike
        The image from which to extract the feature

    channel_map: Dict[str, int]
        A dictionary containing channel names and correspoding indices

    channel: str
        The channel name from which to extract the spherical harmonic
        coefficients

    lmax: int = 4
        See https://github.com/AllenCell/aics-shparam/blob/main/aicsshparam/shparam.py

    sigma: float = 0
        See https://github.com/AllenCell/aics-shparam/blob/main/aicsshparam/shparam.py

    compute_lcc: bool = True
        See https://github.com/AllenCell/aics-shparam/blob/main/aicsshparam/shparam.py

    alignment_2d: bool = False
        See https://github.com/AllenCell/aics-shparam/blob/main/aicsshparam/shparam.py

    make_unique: bool = False
        See https://github.com/AllenCell/aics-shparam/blob/main/aicsshparam/shparam.py

    prefix: Optional[str] = None
        A prefix to prepend to the keys in the result dictionary

    """
    channel = channel_map[channel]
    (coeffs, _), _ = get_shcoeffs(image=img[channel], lmax=lmax,
                                  sigma=sigma, compute_lcc=compute_lcc,
                                  alignment_2d=alignment_2d, make_unique=make_unique)

    if prefix is not None and len(prefix) > 0:
        coeffs = {f"{prefix}_{k}":v for k,v in coeffs.items()}
    return coeffs
