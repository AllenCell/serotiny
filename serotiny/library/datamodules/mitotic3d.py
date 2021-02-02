
# Note - you must have torchvision installed for this example
from torchvision import transforms
from ..image import tiff_loader_CZYX
import numpy as np
from ...constants import DatasetFields
from ...library.data import load_data_loader, Load3DImage, LoadClass, LoadId
from .base_datamodule import BaseDataModule

from aicsimageprocessing.resize import resize_to


class Mitotic3DDataModule(BaseDataModule):

    def __init__(
        self,
        config: dict,
        batch_size: int,
        num_workers: int,
        data_dir: str = './',
    ):

        super().__init__(
            config=config,
            batch_size=batch_size,
            num_workers=num_workers,
            transform_list=[
                transforms.Lambda(lambda x: resize_to(x, (6, 64, 128, 96))),
                transforms.ToTensor()
            ],
            train_transform_list=[
                transforms.Lambda(lambda x: resize_to(x, (6, 64, 128, 96))),
                transforms.ToTensor()
            ],
            x_label="cell_image",
            y_label="mitotic_class",
            data_dir=data_dir,
        )

        self.loaders = {
            # Use callable class objects here because lambdas aren't picklable
            "id": LoadId(self.id_fields),
            self.y_label: LoadClass(len(self.classes)),
            self.x_label: Load3DImage(
                DatasetFields.CellImage3DPath,
                self._num_channels,
                self.channel_indexes,
                self.transform,
            ),
        }

    def load_image(self, dataset):
        return tiff_loader_CZYX(
            path_str=dataset[DatasetFields.CellImage3DPath].iloc[0],
            channel_indexes=self.channel_indexes,
            select_channels=None,
            output_dtype=np.float32,
            channel_masks=None,
            mask_thresh=0
        )

    def get_dims(self, img):
        return (img.shape[1], img.shape[2], img.shape[3])

    def train_dataloader(self):
        train_dataset = self.datasets['train']
        train_loaders = self.loaders.copy()
        train_loaders[self.x_label] = Load3DImage(
            DatasetFields.CellImage3DPath,
            self._num_channels,
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
