from pathlib import Path
from fire import Fire

from .omegaconf_decorator import omegaconf_decorator
from .transform_dataframe_cli import TransformDataframeCLI
from .image_cli import ImageCLI

from serotiny_steps.train_model import train_model
from serotiny_steps.apply_model import apply_model
from serotiny_steps.merge_data import merge_data
from serotiny_steps.partition_data import partition_data
from serotiny_steps.fit_pca import fit_pca
from serotiny_steps.make_aics_mnist_dataset import make_aics_mnist_dataset


def dummy(model_config, datamodule_config, arg1, arg2="abc"):
    print(model_config)
    print(datamodule_config)
    print(arg1)
    print(arg2)


class CLI:
    def __init__(self):
        self.dummy = omegaconf_decorator(dummy, "model_config", "datamodule_config")
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
