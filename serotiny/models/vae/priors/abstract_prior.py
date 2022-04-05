import torch.nn as nn

_full_slice = slice(None, None, None)


class Prior(nn.Module):
    def __init__(self, indices=_full_slice, **kwargs):
        super().__init__()
        self.indices = indices
        pass

    def kl_divergence(self, z):
        raise NotImplementedError

    def sample(self, z):
        raise NotImplementedError
