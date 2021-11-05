from .apply_transforms import transform_images
from .extract_features import extract_features_batch
from .._utils import omegaconf_decorator


class ImageCLI:
    transform = omegaconf_decorator(transform_images)
    extract_features = omegaconf_decorator(extract_features_batch)
