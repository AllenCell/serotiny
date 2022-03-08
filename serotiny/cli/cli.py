import os
import sys
from pathlib import Path

import hydra
from fire import Fire
from omegaconf import OmegaConf
from serotiny.ml_ops import _do_model_op_wrapper
from serotiny.ml_ops.utils import get_serotiny_project

import serotiny.cli.image_cli as image_cli
import serotiny.cli.config_cli as config_cli


def print_help():
    print("Usage:\n  serotiny COMMAND")
    print("\nValid COMMAND values:")
    print("  train - train a model")
    print("  test - test a model")
    print("  config - create a config yaml, given a Python class/function")
    print("  image - image operations")
    print("\nFor more info on each command do:")
    print("  serotiny COMMAND --help")


def main():
    try:
        mode = sys.argv.pop(1)
    except:
        mode = "help"

    if "help" in mode or mode == "-h":
        print_help()
        return

    if mode in ["train", "test"]:

        # hydra modes
        sys.argv[0] += f" {mode}"

        hydra.main(
            config_path=None, config_name=mode
        )(_do_model_op_wrapper)()
    else:
        # fire modes
        sys.argv.insert(1, mode)
        cli_dict = {
            "config": config_cli,
            "image": image_cli,
        }

        if mode in cli_dict:
            Fire(cli_dict)
        else:
            print_help()

if __name__ == "__main__":
    main()

