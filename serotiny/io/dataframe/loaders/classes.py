import torch
from typing import Optional
from .abstract_loader import Loader
import numpy as np

def to_onehot(labels, num_classes):
    labels = labels.type(torch.int64)
    labels_onehot = torch.zeros(1, num_classes).type_as(labels)
    labels_onehot.scatter_(1, labels.view(-1, 1), 1)
    return labels_onehot

class LoadClass(Loader):
    """Loader class, used to retrieve class values from the dataframe,"""

    def __init__(
        self, 
        num_classes: int, 
        y_encoded_label: str, 
        binary: bool = False, 
        dtype: str = "float",
        map_dict: Optional[bool] = None):
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
        self.map_dict = map_dict
        self.dtype = np.dtype(dtype).type

    def __call__(self, row):
        labels = row[self.y_encoded_label]
        if self.map_dict:
            labels = self.map_dict[labels]
        labels = torch.tensor(labels)
        if self.binary:
            labels = to_onehot(labels, self.num_classes)

        labels = self.dtype(labels)
        if len(labels.shape) == 2:
            labels = labels.squeeze(axis=0)
        return labels

class LoadClassWithValues(Loader):
    """Loader one hot classes for a column with values based on another column """

    def __init__(
        self, 
        num_classes: int, 
        y_encoded_label: str, 
        y_value_label: str, 
        dtype: str = "float",
        map_dict: Optional[dict] = None
        ):
        """
        Parameters
        ----------
        num_classes: int
            Number of possible class values

        y_encoded_label: str
            Name of column containing the class

        y_value_label: str
            Name of column containing the value

        """

        super().__init__()
        self.num_classes = num_classes
        self.y_value_label = y_value_label
        self.y_encoded_label = y_encoded_label
        self.map_dict = map_dict
        self.dtype = np.dtype(dtype).type

    def __call__(self, row):
        # import ipdb
        # ipdb.set_trace()
        labels = row[self.y_encoded_label]
        if self.map_dict:
            labels = self.map_dict[labels]
        labels = torch.tensor(labels)
        labels = to_onehot(labels, self.num_classes)
        values = torch.tensor(float(row[self.y_value_label]))
        labels[labels == 1] = values
        if len(labels.shape) == 2:
            labels = labels.squeeze(axis=0)
        return self.dtype(labels)

