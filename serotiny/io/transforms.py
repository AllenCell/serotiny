from typing import Sequence

import math
import numpy as np
import torch
import torch.nn.functional as F
from aicsimageprocessing.resize import resize_to, resize

class ResizeTo:
    def __init__(self, target_dims):
        self.target_dims = target_dims

    def __call__(self, img):
        return resize_to(img, self.target_dims)


class ResizeBy:
    def __init__(self, factor, method="nearest"):
        self.method = method
        self.factor = factor

    def __call__(self, img):
        if isinstance(self.factor, Sequence):
            factor = self.factor
            assert len(factor) == len(img.shape[1:])
        elif isinstance(self.factor, (int, float)):
            n_dims = len(img.shape[1:])
            factor = (1, *(n_dims * [self.factor]))
        else:
            raise TypeError(f"Unexpected factor, of type {type(factor)}")

        return resize(img, factor, self.method)


class Permute:
    def __init__(self, target_dims):
        self.target_dims = target_dims

    def __call__(self, img):
        return torch.tensor(img).permute(*self.target_dims)


class PadTo:
    def __init__(self, target_dims, mode="constant", value=0):
        self.target_dims = target_dims
        self.mode = mode
        self.value = value

    def __call__(self, img):
        img = torch.tensor(img)
        pad = []
        for i, dim in enumerate(self.target_dims):
            pad_dim = (dim - img.shape[i + 1]) / 2

            # when 2 * pad_dim is even, this doesn't change the result.
            # when 2 * pad_dim is odd, this makes padding amount one pixel/voxel
            # bigger on one side
            pad.append(math.floor(pad_dim))
            pad.append(math.ceil(pad_dim))

        # pytorch pad function expects padding amount in reverse order
        pad = pad[::-1]

        return F.pad(img, pad, mode=self.mode, value=self.value)


class MinMaxNormalize:
    def __init__(self, clip_min=None, clip_max=None):
        self.clip_min = clip_min
        self.clip_max = clip_max

    def __call__(self, img):
        img = torch.tensor(img)

        if self.clip_min is not None:
            if isinstance(self.clip_min, (int, float)):
                clip_min = [self.clip_min] * img.shape[0]
            else:
                clip_min = self.clip_min
        else:
            clip_min = [None] * img.shape[0]
        assert len(clip_min) == img.shape[0]

        if self.clip_max is not None:
            if isinstance(self.clip_max, (int, float)):
                clip_max = [self.clip_max] * img.shape[0]
            else:
                clip_max = self.clip_max
        else:
            clip_max = [None] * img.shape[0]
        assert len(clip_max) == img.shape[0]

        for chan in range(img.shape[0]):
            if clip_min[chan] is not None:
                img[chan] = torch.where(img[chan] < clip_min[chan],
                                        torch.tensor(clip_min[chan], dtype=img.dtype),
                                        img[chan])

            if clip_max[chan] is not None:
                img[chan] = torch.where(img[chan] > clip_max[chan],
                                        torch.tensor(clip_max[chan], dtype=img.dtype),
                                        img[chan])

            m = img[chan].min()
            M = img[chan].max()
            img[chan] = (img[chan] - m) / (M - m)

        return img


class CropCenter:
    def __init__(self, cropz, cropx, cropy, pad=2, center_of_mass=None):
        self.cropz = cropz
        self.cropx = cropx
        self.cropy = cropy
        self.pad = pad
        self.center_of_mass = center_of_mass

    def __call__(self, img):
        c,z,x,y = img.shape

        if self.center_of_mass is None:
            center_of_mass = (z // 2, x // 2, y // 2)
        else:
            center_of_mass = self.center_of_mass

        startz = max(0, center_of_mass[0] - (self.cropz // 2) - self.pad)
        startx = max(0, center_of_mass[1] - (self.cropx // 2) - self.pad)
        starty = max(0, center_of_mass[2] - (self.cropy // 2) - self.pad)

        endz = startz + self.cropz + 2 * self.pad
        endx = startx + self.cropx + 2 * self.pad
        endy = starty + self.cropy + 2 * self.pad

        img = img[:,
                  startz: endz,
                  startx: endx,
                  starty: endy]
        return img
