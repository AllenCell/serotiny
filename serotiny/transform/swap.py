import numpy as np


class SwapAxes():
    def __init__(self, first: int, second: int):
        self.first = first
        self.second = second

    def __call__(self, input: np.ndarray):
        return np.swapaxes(input, self.first, self.second)
