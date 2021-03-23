import multiprocessing as mp
import pytorch_lightning as pl

import torch
from torch.utils.data import Dataset, DataLoader

class DummyImageDataset(Dataset):
    """
    Instantiate a dummy pytorch dataset

    Parameters:
    x_label: str
        Key to retrieve image

    y_label: str
        Key to retrieve image label

    dims: str
        Dimensions of image

    length:
        Length of the dummy dataset

    """

    # TODO: Should we include input_channels and output_channels so whatever is returned
    #       will include only the specific channels?
    def __init__(self, x_label, y_label, length, input_dims, output_dims, *args, **kwargs):
        self.length = length
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.x_label = x_label
        self.y_label = y_label

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            self.x_label: torch.randn(self.input_dims),
            self.y_label: torch.randn(self.output_dims),
        }
    
    
def make_dataloader(dataset, batch_size, num_workers):
    """
    Instantiate dummy dataset and return dataloader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=True,
        num_workers=num_workers,
        multiprocessing_context=mp.get_context("fork"),
    )


class DummyImageDatamodule(pl.LightningDataModule):
    """
    A pytorch lightning datamodule that handles the logic for
    loading a dummy dataset

    Parameters
    -----------
    batch_size: int
        batch size for dataloader

    num_workers: int
        Number of worker processes for dataloader

    x_label: str
        x_label key to retrive image

    y_label: str
        y_label key to retrieve image label

    dims: list
        Dimensions for dummy images

    length: int
        Length of dummy dataset

    channels: list = [],
        Number of channels for dummy images
    """

    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        x_label: str,
        y_label: str,
        input_dims: list,
        output_dims: list,
        length: int,
        input_channels: list = [],
        output_channels: list = [],
        **kwargs
    ):

        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.x_label = x_label
        self.y_label = y_label
        self.length = length

        self.num_input_channels = len(input_channels)
        self.num_output_channels = len(output_channels)
        self.input_dims = input_dims
        self.output_dims = output_dims
        
        
    def load_image(self, dataset):
        return dataset[0]

    
    def get_dims(self, img):
        """
        Get dimensions of input image
        """
        return img.shape


    def setup(self, stage=None):
        self.dataset = DummyImageDataset(self.x_label, self.y_label, self.length, self.input_dims, self.output_dims)
        
        self.dataloader = make_dataloader(
            self.dataset,
            self.batch_size,
            self.num_workers,
        )
        
        # Load a test image to get image dimensions after transform
        test_image = self.load_image(self.dataset)

        # Get test image dimensions
        dimensions = self.get_dims(test_image[self.x_label])

        self.dims = dimensions
        
        
    def train_dataloader(self):
        return self.dataloader

    def val_dataloader(self):
        return self.dataloader

    def test_dataloader(self):
        return self.dataloader
