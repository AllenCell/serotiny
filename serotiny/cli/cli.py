import sys

from serotiny.ml_ops import _do_model_op_wrapper
from serotiny.ml_ops.utils import get_serotiny_project


def print_help():
    from textwrap import dedent

    print(
        dedent(
            """
    Usage:
      serotiny COMMAND

    Valid COMMAND values:
      train - train a model
      test - test a model
      predict - use a trained model to output predictions
      config - create a config yaml, given a Python class/function
      dataframe - utils to manipulate .csv dataframes
      image - image operations

    For more info on each command do:")
      serotiny COMMAND --help")
    """
        ).strip()
    )


def main():
    if sys.argv[0].endswith("serotiny"):
        try:
            mode = sys.argv.pop(1)
        except:
            mode = "help"
    elif sys.argv[0].endswith("serotiny.train"):
        mode = "train"
    elif sys.argv[0].endswith("serotiny.test"):
        mode = "test"
    elif sys.argv[0].endswith("serotiny.predict"):
        mode = "predict"
    else:
        raise NotImplementedError(f"Unknown command: '{sys.argv[0]}")

    if "help" in mode or mode == "-h":
        print_help()
        return

    # hydra modes
    if mode in ["train", "test", "predict"]:
        import hydra

        if sys.argv[0].endswith("serotiny"):
            sys.argv[0] += f".{mode}"

        hydra.main(config_path=None, config_name=mode)(_do_model_op_wrapper)()

    # fire modes
    else:
        import serotiny.cli.image_cli as image_cli
        import serotiny.cli.config_cli as config_cli
        from serotiny.cli.dataframe_cli import DataframeTransformCLI as dataframe_cli
        from fire import Fire

        sys.argv.insert(1, mode)
        cli_dict = {
            "config": config_cli,
            "image": image_cli,
            "dataframe": dataframe_cli,
        }

        if mode in cli_dict:
            Fire(cli_dict)
        else:
            print_help()


if __name__ == "__main__":
    main()
