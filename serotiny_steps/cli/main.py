from fire import Fire


class CLI:
    @classmethod
    def model(cls):
        """
        Model training and inference commands
        """
        from .model_cli import ModelCLI
        ModelCLI.__doc__ = cls.model.__doc__
        return ModelCLI

    @classmethod
    def image(cls):
        """
        Image transforms and feature extraction commands
        """
        from .image_cli import ImageCLI
        ImageCLI.__doc__ = cls.image.__doc__
        return ImageCLI

    @classmethod
    def dataframe(cls):
        """
        Dataframe wrangling commands
        """
        from .dataframe_cli import DataframeCLI
        DataframeCLI.__doc__ = cls.dataframe.__doc__
        return DataframeCLI

    @classmethod
    def misc(cls):
        """
        Miscellaneous commands
        """
        from .misc_cli import MiscCLI
        MiscCLI.__doc__ = cls.misc.__doc__
        return MiscCLI

def main():
    Fire(CLI)

if __name__ == "__main__":
    main()
