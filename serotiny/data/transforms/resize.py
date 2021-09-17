from typing import Sequence

import math
import numpy as np
import torch

from aicsimageprocessing.resize import resize_to, resize
from .pad import PadTo


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


class CropCenter:
    def __init__(self, cropz, cropx, cropy, pad=0, center_of_mass=None,
                 force_size=True):
        self.cropz = cropz + (cropz % 2 != 0)
        self.cropx = cropx + (cropx % 2 != 0)
        self.cropy = cropy + (cropy % 2 != 0)

        self.pad = pad
        self.center_of_mass = center_of_mass
        self.force_size = force_size

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

        if self.force_size:
            pad_to = PadTo(target_dims=[self.cropz, self.cropx, self.cropy])
            img = pad_to(img)

        return img
