from typing import Sequence, Union, Callable
from torchvision import transforms
from ...image import png_loader
from ...data import load_data_loader
from ...data.loaders import Load2DImage, LoadClass, LoadColumns
from ..constants import DatasetFields
from ..base_datamodule import BaseDataModule


class ACTK2DDataModule(BaseDataModule):
    """
    A pytorch lightning datamodule that handles the logic for
    loading 2D ACTK images

    Parameters
    -----------
    x_label: str
        Column name used to load an image (x)

    y_label: str
        Column name used to load the image label (y)

    batch_size: int
        Batch size for the dataloader

    num_workers: int
        Number of worker processes to create in dataloader

    id_fields: Sequence[Union[str, int]]
        Id column name for loader

    channels: Sequence[Union[str, int]]
        List of channels in the images

    select_channels: Sequence[Union[str, int]]
        List of channels to subset the original channel list

    data_dir: str
        Path to data folder containing csv's for train, val,
        and test splits

    resize_to: int
        Resize input images to a square of this size

    encoded_label_suffix: str
        a column of categorical variables is converted into an integer
        representation. This column in named
        encoded_label + encoded_label_suffix
        Example:
            encoded_label = "ChosenMitoticClass"
            encoded_label_suffix = "Integer"

    classes: list
        List of classes in the encoded_label column
    """

    def __init__(
        self,
        transform_list: Sequence[Callable],
        train_transform_list: Sequence[Callable],
        x_label: str,
        y_label: str,
        batch_size: int,
        num_workers: int,
        id_fields: Sequence[Union[str, int]],
        channels: Sequence[Union[str, int]],
        select_channels: Sequence[Union[str, int]],
        encoded_label_suffix: str,
        classes: list,
        data_dir: str,
        **kwargs,
    ):

        self.classes = classes

        super().__init__(
            channels=channels,
            select_channels=select_channels,
            batch_size=batch_size,
            num_workers=num_workers,
            transform_list=[
                transforms.ToPILImage(),
                transforms.Resize(resize_to),
                transforms.CenterCrop(resize_to),
                transforms.ToTensor(),
            ],
            train_transform_list=[
                transforms.ToPILImage(),
                transforms.Resize(resize_to),
                transforms.CenterCrop(resize_to),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
            ],
            x_label=x_label,
            y_label=y_label,
            data_dir=data_dir,
            **kwargs,
        )

        self.x_label = x_label
        self.y_label = y_label
        self.y_encoded_label = y_label + encoded_label_suffix

        self.loaders = {
            # Use callable class objects here because lambdas aren't picklable
            "id": LoadColumns(self.id_fields),
            self.y_label: LoadClass(len(self.classes), self.y_encoded_label),
            self.x_label: Load2DImage(
                DatasetFields.Chosen2DProjectionPath,
                self.num_channels,
                self.channel_indexes,
                self.transform,
            ),
        }

    def load_image(self, dataset):
        """
        Load a single 2D image given a path
        """
        return png_loader(
            dataset[DatasetFields.Chosen2DProjectionPath].iloc[0],
            channel_order="CYX",
            indexes={"C": self.channel_indexes or range(self.num_channels)},
            transform=self.transform,
        )

    def get_dims(self, img):
        """
        Get dimensions of input image
        """
        return (img.shape[1], img.shape[2])

    def train_dataloader(self):
        """
        Instantiate train dataloader.
        """
        train_dataset = self.datasets["train"]
        train_loaders = self.loaders.copy()
        train_loaders[self.x_label] = Load2DImage(
            DatasetFields.Chosen2DProjectionPath,
            self.num_channels,
            self.channel_indexes,
            self.train_transform,
        )
        train_dataloader = load_data_loader(
            train_dataset,
            train_loaders,
            transform=self.train_transform,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        return train_dataloader
