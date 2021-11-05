from fire import Fire


class CLI:
    @classmethod
    def model(cls):
        from .model_cli import ModelCLI
        return ModelCLI

    @classmethod
    def image(cls):
        from .image_cli import ImageCLI
        return ImageCLI

    @classmethod
    def dataframe(cls):
        from .dataframe_cli import DataframeCLI
        return DataframeCLI

    @classmethod
    def misc(cls):
        from .misc_cli import MiscCLI
        return MiscCLI

def main():
    Fire(CLI)

if __name__ == "__main__":
    main()
