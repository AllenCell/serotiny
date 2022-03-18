import math

import numpy as np
import torch
import torch.nn.functional as F


def split_number(n):
    middle = n / 2
    bottom = math.floor(middle)
    top = math.ceil(middle)
    return (bottom, top)


def pull_to(array, axes: int):
    if not isinstance(array, np.ndarray):
        array = np.array(array)
    missing = axes - len(array.shape)
    pull = np.expand_dims(array, tuple(range(missing)))
    return pull


def expand_to(array, dimensions, pad=None):
    pad = pad or {}
    in_shape = array.shape
    in_dimensions = len(in_shape)
    missing = len(dimensions) - in_dimensions
    pull = np.expand_dims(array, tuple(range(missing)))
    around = [
        split_number(dimensions[dimension] - pull.shape[dimension])
        for dimension in range(in_dimensions)
    ]
    expand = np.pad(pull, around, **pad)
    return expand


def to_tensor(value):
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    else:
        return torch.Tensor([value])


def expand_columns(rows, expanded_columns, dimensions, pad=None):
    if not rows:
        return []

    first_row = rows[0]
    all_columns = list(first_row.keys())
    collated = {column: [] for column in all_columns}

    for row in rows:
        for column in all_columns:
            value = row[column]
            if column in expanded_columns:
                value = expand_to(value, dimensions, pad)
            tensor = to_tensor(value)
            collated[column].append(tensor)

    for column in all_columns:
        collated[column] = torch.stack(collated[column])

    return collated


class ExpandTo:
    def __init__(self, dimensions, pad=None):
        self.dimensions = dimensions
        self.pad = pad or {}

    def __call__(self, in_array: np.ndarray):
        return expand_to(in_array, self.dimensions, self.pad)


class ExpandColumns:
    def __init__(self, columns, dimensions, pad=None):
        self.columns = columns
        self.dimensions = dimensions
        self.pad = pad or {}

    def __call__(self, rows):

        return expand_columns(rows, self.columns, self.dimensions, self.pad)


class PadTo:
    def __init__(self, target_dims, mode="constant", value=0):
        self.target_dims = target_dims
        self.mode = mode
        self.value = value

    def __call__(self, img):
        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img)

        pad = []
        for i, dim in enumerate(self.target_dims):
            pad_dim = (dim - img.shape[i + 1]) / 2
            pad_dim = max(0, pad_dim)

            # when 2 * pad_dim is even, this doesn't change the result.
            # when 2 * pad_dim is odd, this makes padding amount one pixel/voxel
            # bigger on one side
            pad.append(math.floor(pad_dim))
            pad.append(math.ceil(pad_dim))

        # pytorch pad function expects padding amount in reverse order
        pad = pad[::-1]

        return F.pad(img, pad, mode=self.mode, value=self.value)
