import torch


class Project:
    def __init__(self, axis, mode="max"):
        self.axis = axis
        self.mode = mode

    def __call__(self, img):
        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img)

        axis = {"z": -3, "y": -2, "x": -1}

        if self.axis == "z":
            assert len(img.shape) >= 3

        if self.mode == "max":
            return img.max(axis=axis[self.axis])[0]
        elif self.mode == "mean":
            return img.mean(axis=axis[self.axis])
        elif self.mode == "median":
            return img.median(axis=axis[self.axis])
        else:
            raise NotImplementedError
