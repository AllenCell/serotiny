from typing import Sequence

import numpy as np

from .abstract_loader import Loader


class LoadRandomTensor(Loader):
    """Loader class, used to load a dummy tensor, generated randomly.

    It ignores the underlying dataframe
    """

    def __init__(
        self,
        column: str,
        dims: Sequence[int],
    ):
        """
        Parameters
        ----------
        column: str
            Ignored. Kept for consistency with other loaders

        dims:
            Tensor dimensions
        """

        super().__init__()
        self.column = column
        self.dims = dims

    def __call__(self, row):
        return np.random.randn(*self.dims)
