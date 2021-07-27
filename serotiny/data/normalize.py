import numpy as np


class NormalizeAbsolute():
    def __init__(self, axis=-1, order=2):
        self.axis = axis
        self.order = order

    def __call__(self, a):
        l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
        l2[l2==0] = 1
        return a / np.expand_dims(l2, axis)    


class NormalizeMean():
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


class NormalizeMinMax():
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
        ar = 2*(ar - norm_min) / (norm_max - norm_min)-1

        return ar.astype(np.float32)
