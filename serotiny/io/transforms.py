import torch
from aicsimageprocessing.resize import resize_to

class ResizeTo:
    def __init__(self, target_dims):
        self.target_dims = target_dims

    def __call__(self, img):
        return resize_to(img, self.target_dims)

class Permute:
    def __init__(self, target_dims):
        self.target_dims = target_dims

    def __call__(self, img):
        return torch.tensor(img).permute(*self.target_dims)
