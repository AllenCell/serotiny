import numpy as np
import torch


class NormalizeAbsolute:
    def __init__(self, axis=-1, order=2):
        self.axis = axis
        self.order = order

    def __call__(self, a):
        l2 = np.atleast_1d(np.linalg.norm(a, self.order, self.axis))
        l2[l2 == 0] = 1
        return a / np.expand_dims(l2, self.axis)


class NormalizeMean:
    def __init__(self, scale=1.0):
        self.scale = scale

    def __call__(self, img, per_dim=None):
        """Subtract mean, set STD to 1.0.

        Parameters:
          per_dim: normalize along other axes dimensions not equal to per dim
        """
        axis = tuple([i for i in range(img.ndim) if i != per_dim])
        slices = tuple(
            [slice(None) if i == per_dim else np.newaxis for i in range(img.ndim)]
        )  # to handle broadcasting
        result = img.astype(np.float32)
        result -= np.mean(result, axis=axis)[slices]
        result /= np.std(result, axis=axis)[slices]
        result *= self.scale
        return result


class NormalizeMinMax:
    def __init__(
        self, clip_min=None, clip_max=None, clip_quantile=False, return_torch=False
    ):
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.clip_quantile = clip_quantile
        self.return_torch = return_torch

    def __call__(self, img):
        if self.clip_min is not None:
            if isinstance(self.clip_min, (int, float)):
                if not self.clip_quantile:
                    clip_min = [self.clip_min] * img.shape[0]
                else:
                    clip_min = [
                        np.quantile(img[ch], self.clip_min)
                        for ch in range(img.shape[0])
                    ]
            else:
                clip_min = self.clip_min
        else:
            clip_min = [None] * img.shape[0]
        assert len(clip_min) == img.shape[0]

        if self.clip_max is not None:
            if isinstance(self.clip_max, (int, float)):
                if not self.clip_quantile:
                    clip_max = [self.clip_max] * img.shape[0]
                else:
                    clip_max = [
                        np.quantile(img[ch], self.clip_max)
                        for ch in range(img.shape[0])
                    ]
            else:
                clip_max = self.clip_max
        else:
            clip_max = [None] * img.shape[0]
        assert len(clip_max) == img.shape[0]

        for chan in range(img.shape[0]):
            if clip_min[chan] is not None:
                _chan_result = np.where(
                    img[chan] < clip_min[chan],
                    clip_min[chan],
                    img[chan],
                )
                if isinstance(img, torch.Tensor):
                    _chan_result = torch.tensor(_chan_result)
                img[chan] = _chan_result

            if clip_max[chan] is not None:
                _chan_result = np.where(
                    img[chan] > clip_max[chan],
                    clip_max[chan],
                    img[chan],
                )
                if isinstance(img, torch.Tensor):
                    _chan_result = torch.tensor(_chan_result)
                img[chan] = _chan_result

            m = img[chan].min()
            M = img[chan].max()
            img[chan] = (img[chan] - m) / (M - m)

        if self.return_torch:
            img = torch.tensor(img)
        return img
