from fire import Fire

from serotiny_steps.train_model import train_model
from serotiny_steps.apply_model import apply_model
from serotiny_steps.merge_data import merge_data
from serotiny_steps.partition_data import partition_data
from serotiny_steps.fit_pca import fit_pca
from serotiny_steps.make_aics_mnist_dataset import make_aics_mnist_dataset
from .omegaconf_decorator import omegaconf_decorator
from .transform_dataframe_cli import TransformDataframeCLI
from .image_cli import ImageCLI


class CLI:
    def __init__(self):
        self.model = {
            "train": omegaconf_decorator(
                train_model,
                "model",
                "datamodule",
                "trainer",
                "model_zoo",
                "loggers",
                "callbacks",
            ),
            "predict": omegaconf_decorator(
                apply_model,
                "datamodule",
                "trainer",
                "model_zoo",
                "loggers",
                "callbacks",
            ),
        }
        self.dataframe = {
            "transform": TransformDataframeCLI,
            "merge": merge_data,
            "partition": partition_data,
        }

        self.image = ImageCLI
        self.utils = {
            "fit_pca": fit_pca,
            "make_aics_mnist_dataset": make_aics_mnist_dataset,
        }
        # self.feature_extraction = FeatureExtractionCLI()


def main():
    Fire(CLI())


if __name__ == "__main__":
    main()
