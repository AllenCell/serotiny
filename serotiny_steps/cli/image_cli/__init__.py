from ..utils.base_cli import BaseCLI


class ImageCLI(BaseCLI):
    @classmethod
    def transform(cls):
        """
        Transform images given in a manifest, using transforms given by
        a config
        """
        from .apply_transforms import transform_images
        return cls._decorate(transform_images)

    @classmethod
    def extract_features(cls):
        """
        Extract features from an image in a dataframe row, using extractors given by
        a config
        """
        from .extract_features import extract_features_batch
        return cls._decorate(extract_features_batch)
