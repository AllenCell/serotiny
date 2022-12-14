from fire import Fire
from serotiny.cli.dataframe_cli import DataframeTransformCLI as dataframe_cli


def config_cli():
    raise NotImplementedError


def image_cli():
    raise NotImplementedError


if __name__ == "__main__":
    cli_dict = {
        "config": config_cli,
        "image": image_cli,
        "dataframe": dataframe_cli,
    }

    Fire(cli_dict)
