import os
from typing import Callable, Optional, Sequence, Union, Type

import numpy as np

from serotiny.io.image import image_loader

from .abstract_loader import Loader


class LoadImage(Loader):
    """Loader class, used to retrieve images from paths given in a dataframe
    column."""

    def __init__(
        self,
        column: str,
        file_type: str = "tiff",
        select_channels: Optional[Sequence] = None,
        transform: Optional[Callable] = None,
        reader: Optional[str] = None,
        dtype: Optional[Union[str, Type[np.number]]] = None,
        load_as_torch: bool = True,
        use_cache: bool = False,
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

        transform: Optional[Callable] = None
            Transform to apply upon loading the image.

        reader: Optional[str] = None
            `aicsimageio` reader to use

        dtype: np.dtype = np.float32
            dtype to use. see https://numpy.org/doc/stable/reference/generated/numpy.ndarray.astype.html  # noqa
            for more info on this

        load_as_torch: bool = True
            Whether to load the image as a torch tensor rather than a numpy
            array. Some transforms require this.

        use_cache: bool = True
            Whether to cache images after downloading them once

        """
        super().__init__()
        self.column = column
        self.select_channels = select_channels
        self.reader = reader
        self.load_as_torch = load_as_torch

        if file_type not in ("tiff"):
            raise NotImplementedError(f"File type {file_type} not supported.")

        self.file_type = file_type
        self.dtype = dtype
        self.transform = transform
        self.use_cache = use_cache

    def _get_cached_path(self, path):
        if not self.use_cache:
            return path

        conf_path = os.getenv("FSSPEC_CONFIG_DIR")
        if conf_path is not None:
            if "simplecache::" not in str(path):
                return "simplecache::" + path
        return path

    def __call__(self, row):
        if self.file_type == "tiff":
            return image_loader(
                self._get_cached_path(row[self.column]),
                select_channels=self.select_channels,
                output_dtype=self.dtype,
                transform=self.transform,
                reader=self.reader,
                return_as_torch=self.load_as_torch,
            )
