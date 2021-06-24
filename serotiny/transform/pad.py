import math
import numpy as np
import torch


def split_number(n):
    middle = n / 2
    bottom = math.floor(middle)
    top = Math.ceil(middle)
    return (bottom, top)


def expand_to(array, dimensions, pad=None):
    pad = pad or {}
    in_shape = array.shape
    in_dimensions = len(in_shape)
    missing = len(dimensions) - in_dimensions
    pull = np.expand_dims(array, tuple(range(missing)))
    around = [
        split_number(dimensions[dimension] - pull.shape[dimension])
        for dimension in range(in_dimensions)]
    expand = np.pad(pull, around, **pad)
    return expand
    

def expand_columns(rows, expanded_columns, dimensions, pad=None):
    if not rows:
        return []

    first_row = rows[0]
    all_columns = list(first_row.keys())
    unexpanded_columns = set(all_columns) - set(expanded_columns)
    collated = {
        column: []
        for column in all_columns}

    for row in rows:
        for column in all_columns:
            value = row[column]
            if column in expanded_columns:
                value = expand_to(value, dimensions, pad)
            collated[column].append(value)

    for column in all_columns:
        collated[column] = torch.stack(collated[column])

    return tuple(collated.values())


class ExpandTo():
    def __init__(
            self,
            dimensions,
            pad=None):
        self.dimensions = dimensions
        self.pad = pad or {}

    def __call__(self, in_array: np.ndarray):
        return expand_to(in_array, self.dimensions, self.pad)


class ExpandColumns():
    def __init__(
            self,
            columns,
            dimensions,
            pad=None):
        self.columns = columns
        self.dimensions = dimensions
        self.pad = pad or {}

    def __call__(
            self,
            rows):

        return expand_columns(
            rows,
            self.columns,
            self.dimensions,
            self.pad)
