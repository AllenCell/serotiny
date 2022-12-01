import torch

from .abstract_loader import Loader


class LoadClass(Loader):
    """Loader class, used to retrieve class values from the dataframe,"""

    def __init__(self, num_classes: int, y_encoded_label: str, binary: bool = False):
        """
        Parameters
        ----------
        num_classes: int
            Number of possible class values

        y_encoded_label: str
            Name of column containing the class value

        binary: bool
            Flag to determine whether to return class values or
            one-hot vectors

        """

        super().__init__()
        self.num_classes = num_classes
        self.binary = binary
        self.y_encoded_label = y_encoded_label

    def __call__(self, row):
        if self.binary:
            return torch.tensor([row[str(i)] for i in range(self.num_classes)])

        return torch.tensor(row[self.y_encoded_label])
