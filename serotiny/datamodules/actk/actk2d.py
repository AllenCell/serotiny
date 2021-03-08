# Note - you must have torchvision installed for this example
from torchvision import transforms
from ...image import png_loader
from ...data import load_data_loader
from ...data.loaders import Load2DImage, LoadClass, LoadColumns
from ..constants import DatasetFields
from ..base_datamodule import BaseDataModule
from ..utils import subset_channels


class ACTK2DDataModule(BaseDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        x_label: str,
        y_label: str,
        data_dir: str,
        resize_to: int,
        encoded_label_suffix: str,
        channels: list,
        select_channels: list,
        classes: list,
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
        return png_loader(
            dataset[DatasetFields.Chosen2DProjectionPath].iloc[0],
            channel_order="CYX",
            indexes={"C": self.channel_indexes or range(self.num_channels)},
            transform=self.transform,
        )

    def get_dims(self, img):
        return (img.shape[1], img.shape[2])

    def train_dataloader(self):
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
