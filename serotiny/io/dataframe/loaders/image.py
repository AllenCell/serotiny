from typing import Callable, Optional, Sequence, Union

from omegaconf import ListConfig
import numpy as np

from serotiny.io.image import image_loader

from .abstract_loader import Loader


class LoadImage(Loader):
    """Loader class, used to retrieve images from paths given in a dataframe column."""

    def __init__(
        self,
        column: str,
        file_type: str = "tiff",
        select_channels: Optional[Sequence] = None,
        transforms: Optional[Union[Sequence, Callable]] = None,
        reader: str = "aicsimageio.readers.ome_tiff_reader.OmeTiffReader",
        dtype: np.dtype = np.float32,
    ):
        """
        Parameters
        ----------
        column: str
            Dataframe column which contains the image path

        file_type: str
            File format of the image. For now, "tiff" is the
            only supported format. But in the future other formats (like zarr)
            might be supported too

        select_channels: Optional[Sequence] = None
            List of channels to include in the loaded image.

        transforms: Optional[Union[Sequence, Callable]] = None
            Transform, or list of transforms to apply upon loading the image.

        reader: str = "aicsimageio.readers.ome_tiff_reader.OmeTiffReader"
            `aicsimageio` reader to use

        dtype: np.dtype = np.float32
            dtype to use

        """
        super().__init__()
        self.column = column
        self.select_channels = select_channels
        self.reader = reader

        if file_type not in ("tiff"):
            raise NotImplementedError(f"File type {file_type} not supported.")

        self.file_type = file_type
        self.dtype = dtype

        if isinstance(transforms, (list, tuple, ListConfig)):
            from torchvision.transforms import Compose

            transforms = Compose(transforms)

        self.transforms = transforms

    def __call__(self, row):
        if self.file_type == "tiff":
            return image_loader(
                row[self.column],
                select_channels=self.select_channels,
                output_dtype=self.dtype,
                transform=self.transforms,
                reader=self.reader,
            )
