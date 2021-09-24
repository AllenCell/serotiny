import numpy as np


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
        """Subtract mean, set STD to 1.0

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


class MinMaxNormalize:
    def __init__(self, clip_min=None, clip_max=None, clip_quantile=False):
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.clip_quantile = clip_quantile

    def __call__(self, img):
        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img)

        if self.clip_min is not None:
            if isinstance(self.clip_min, (int, float)):
                if not self.clip_quantile:
                    clip_min = [self.clip_min] * img.shape[0]
                else:
                    clip_min = [
                        img[ch].quantile(self.clip_min).item()
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
                        img[ch].quantile(self.clip_max).item()
                        for ch in range(img.shape[0])
                    ]
            else:
                clip_max = self.clip_max
        else:
            clip_max = [None] * img.shape[0]
        assert len(clip_max) == img.shape[0]

        for chan in range(img.shape[0]):
            if clip_min[chan] is not None:
                img[chan] = torch.where(
                    img[chan] < clip_min[chan],
                    torch.tensor(clip_min[chan], dtype=img.dtype),
                    img[chan],
                )

            if clip_max[chan] is not None:
                img[chan] = torch.where(
                    img[chan] > clip_max[chan],
                    torch.tensor(clip_max[chan], dtype=img.dtype),
                    img[chan],
                )

            m = img[chan].min()
            M = img[chan].max()
            img[chan] = (img[chan] - m) / (M - m)

        return img
