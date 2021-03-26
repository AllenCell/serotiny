from pathlib import Path

from .loaders import (
    Load2DImage,
    Load3DImage,
)

def infer_extension_loader(extension):
    if extension == ".png":
        return Load2DImage(
            chosen_col="true_paths",
            num_channels=3,
            channel_indexes=[0,1,2],
            transform=None
        )

    if extension == ".tiff":
        return Load3DImage(
            chosen_col="true_paths",
        )

    raise NotImplemented(f"Can't determine appropriate loader for given extension {extension}")
