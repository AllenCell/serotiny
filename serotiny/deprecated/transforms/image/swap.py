import numpy as np
import torch


class SwapAxes:
    def __init__(self, first: int, second: int):
        self.first = first
        self.second = second

    def __call__(self, input: np.ndarray):
        print(f"swapping axes {self.first} and {self.second}: {input.shape}")
        output = np.swapaxes(input, self.first, self.second)
        print(f"swapping axes result: {output.shape}")
        return output


class Permute:
    def __init__(self, target_dims):
        self.target_dims = target_dims

    def __call__(self, img):
        return torch.tensor(img).permute(*self.target_dims)
