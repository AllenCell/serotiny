from aicsshparam.shtools import align_image_2d as _align_image_2d


def align_image_2d(img, channel, adjust_for_skew):
    return _align_image_2d(img, channel, make_unique=adjust_for_skew,
                           compute_aligned_image=True)[0]
