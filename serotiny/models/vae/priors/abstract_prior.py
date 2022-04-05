import torch.nn as nn

_full_slice = slice(None, None, None)


class Prior(nn.Module):
    def __init__(self, indices=None, **kwargs):
        super().__init__()
        if indices is None:
            indices = _full_slice
        self.indices = indices

    def kl_divergence(self, z):
        raise NotImplementedError

    def sample(self, z):
        raise NotImplementedError
