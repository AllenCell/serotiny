import os
import sys

import hydra
from omegaconf import OmegaConf
from serotiny.models.ops import train, apply


def main():
    try:
        mode = sys.argv.pop(1)
    except:
        mode = "help"

    if "help" in mode or mode == "-h":
        print("Usage:\n  serotiny COMMAND")
        print("\nValid COMMAND values:")
        print("  train - train a model")
        print("  apply - apply a model")
        print("\nFor more info on each command do:")
        print("  serotiny COMMAND --help")
        return


    if mode not in ["train", "apply"]:
        raise ValueError(
            f"`mode` must be either 'train' or 'test'. Got '{mode}'"
        )

    #if not any(["output_root=" in arg for arg in sys.argv]):
    #    sys.argv.append(f"output_root={os.getcwd()}/output")


    if mode == "train":
        hydra.main(config_path=None,
                   config_name="train")(train)()
    else:
        hydra.main(config_path=None,
                   config_name="apply")(apply)()


if __name__ == "__main__":
    main()

