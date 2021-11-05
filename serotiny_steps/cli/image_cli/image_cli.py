from .apply_transforms import transform_images
from .extract_features import extract_features_batch
from ..omegaconf_decorator import omegaconf_decorator


class ImageCLI:
    def __init__(self):
        self.transform = omegaconf_decorator(transform_images)
        self.extract_features = omegaconf_decorator(extract_features_batch)
