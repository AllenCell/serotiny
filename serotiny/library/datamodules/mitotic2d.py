import pytorch_lightning as pl

# Note - you must have torchvision installed for this example
from torchvision import transforms
from os import listdir
from ..csv import load_csv
from ..image import png_loader
from pathlib import Path
from ...constants import DatasetFields
from ...library.data import load_data_loader, LoadImage, LoadClass, LoadId
from .base_datamodule import BaseDataModule

class Mitotic2DDataModule(BaseDataModule):

    def __init__(
        self,
        config: dict,
        x_label: str,
        y_label: str,
        batch_size: int,
        num_workers: int,
        data_dir: str = './',
    ):

        super().__init__(
            transform_list=[
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
            ],
            train_transform_list=[
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
            ],
            x_label="projection_image",
            y_label="mitotic_class"
        )

        self.x_label = "projection_image"
        self.y_label = "mitotic_class"

        self.loaders = {
            # Use callable class objects here because lambdas aren't picklable
            "id": LoadId(self.id_fields),
            self.y_label: LoadClass(len(self.classes)),
            self.x_label: LoadImage(
                DatasetFields.Chosen2DProjectionPath,
                self._num_channels,
                self.channel_indexes,
                self.transform,
            ),
        }
        
        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        self.dims = (1, 28, 28)

    def load_image(self, dataset):
        return png_loader(
            dataset[DatasetFields.Chosen2DProjectionPath].iloc[0],
            channel_order="CYX",
            indexes={"C": self.channel_indexes or range(self._num_channels)},
            transform=self.transform,
        )

        pass

    def get_dims(self, img):
        return (img.shape[1], img.shape[2])
