from contextlib import suppress
from pathlib import Path
import nbformat
import nbformat.v4 as v4
from omegaconf import OmegaConf


def get_serotiny_project():
    with suppress():
        if (Path.cwd() / ".serotiny").exists():
            with open(".serotiny", "r") as f:
                project_name = f.read().strip()
            return project_name

    return "serotiny"


def flatten_config(cfg):
    import pandas as pd

    conf = pd.json_normalize(cfg, sep="/").to_dict(orient="records")[0]
    keys = list(conf.keys())

    for k in keys:
        with suppress():
            sub_conf = flatten_config(conf[k])
            conf.update({f"{k}/{_k}": v for _k, v in sub_conf.items()})
            del conf[k]
            continue

        if isinstance(conf[k], list):
            for i, el in enumerate(conf[k]):
                try:
                    sub_conf = flatten_config(el)
                    conf.update({f"{k}/{_k}": v for _k, v in sub_conf.items()})
                except:  # noqa
                    conf[f"{k}/{i}"] = el
            del conf[k]

    return pd.json_normalize(conf, sep="/").to_dict(orient="records")[0]


def make_notebook(cfg, path):
    cells = []

    cells.append(
        (
            "code",
            (
                "import yaml"
                "import pytorch_lightning as pl"
                "import torch"
                "import torch.nn as nn"
                "from hydra.utils import instantiate"
            ),
        )
    )

    for subconfig in ("data", "model", "trainer"):
        cells.extend(
            (
                (
                    "markdown",
                    (
                        f"\nBelow is the `{subconfig}` config and instantiation. Edit the yaml code"  # noqa
                        f"\nto change the configuration."
                    ),
                ),
                (
                    "code",
                    (
                        f"{subconfig} = instantiate(yaml.full_load('''\n"
                        + OmegaConf.to_yaml(cfg[subconfig])
                        + "\n"
                        + "'''))\n"
                    ),
                ),
            )
        )

    cells.extend(
        (
            (
                "markdown",
                (
                    "\n---"
                    "\nThe cell below loads a batch from the train dataloader, and prints the"  # noqa
                    "\navailable keys."
                ),
            ),
            (
                "code",
                (
                    "\ntrain_dl = data.train_dataloader()"
                    "\ntrain_batch = next(iter(train_dl))"
                    "\n\nprint(train_batch.keys())"
                ),
            ),
        )
    )

    cells = [
        (
            v4.new_code_cell(source)
            if cell_type == "code"
            else v4.new_markdown_cell(source)
        )
        for cell_type, source in cells
    ]

    notebook = v4.new_notebook(cells=cells)
    nbformat.write(notebook, path)


def save_model_predictions(model, preds, output_dir):
    if hasattr(model, "save_predictions"):
        model.save_predictions(preds, output_dir)
    else:
        import joblib

        joblib.dump(preds, output_dir / "predictions.joblib.xz")
