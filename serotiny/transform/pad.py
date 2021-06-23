import math
import numpy as np

def split_number(n):
    middle = n / 2
    bottom = math.floor(middle)
    top = Math.ceil(middle)
    return (bottom, top)

class ExpandTo():
    def __init__(
            self,
            dimensions,
            pad=None):
        self.dimensions = dimensions
        self.pad = pad or {}

    def __call__(self, in_array: np.ndarray):
        in_shape = in_array.shape
        in_dimensions = len(in_shape)
        missing = len(self.dimensions) - in_dimensions
        pull = np.expand_dims(in_array, tuple(range(missing)))
        around = [
            split_number(self.dimensions[dimension] - pull.shape[dimension])
            for dimension in range(in_dimensions)]
        expand = np.pad(pull, around, **pad)
        return expand
