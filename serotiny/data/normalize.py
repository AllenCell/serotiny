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


class NormalizeMinMax:
    def __init__(self, z_center=None):
        self.z_center = z_center

    def __call__(self, ar: np.ndarray):
        """Returns normalized version of input array.
        The array will be normalized from [min, max] to [-1, 1] linearly
        Parameters
        ----------
        ar
        Input 3d array to be normalized.
        Returns
        -------
        np.ndarray
        Nomralized array, dtype = float32
        """

        ar = ar.astype(np.float32)
        norm_min = ar.min()
        norm_max = ar.max()
        ar = 2 * (ar - norm_min) / (norm_max - norm_min) - 1

        return ar.astype(np.float32)


class NormalizeAroundCenter():
    def __init__(self, scope=32, z_center=None):
        self.scope = scope
        self.z_center = z_center

    def __call__(self, ar: np.ndarray):
        """Returns normalized version of input array.

        The array will be normalized with respect to the mean, std pixel intensity
        of the sub-array of length `self.scope` in the z-dimension centered around the
        array's "z_center".

        Parameters
        ----------
        ar
            Input 3d array to be normalized.
        z_center
            Z-index of cell centers.

        Returns
        -------
        np.ndarray
           Normalized array, dtype = float32

        """

        original_dims = ar.ndim
        if ar.ndim > 3:
            ar = np.squeeze(ar)

        if ar.ndim != 3:
            raise ValueError("Input array must have 3 or more significant dimensions")
        if ar.shape[0] < self.scope:
            raise ValueError("Input array must be at least length 32 in first dimension")
        if self.z_center is None:
            z_center = ar.shape[0] // 2
        else:
            z_center = self.z_center

        z_start = z_center - self.scope // 2
        if z_start < 0:
            z_start = 0
            logger.warn(f"Warning: z_start set to {z_start}")
        if (z_start + self.scope) > ar.shape[0]:
            z_start = ar.shape[0] - self.scope
            logger.warn(f"Warning: z_start set to {z_start}")
        chunk = ar[z_start : z_start + self.scope, :, :]
        ar = ar - chunk.mean()
        ar = ar / chunk.std()

        while original_dims > ar.ndim:
            ar = np.expand_dims(ar, axis=0)

        return ar.astype(np.float32)



