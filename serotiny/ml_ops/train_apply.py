import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import OmegaConf
import copy

from .mlflow_utils import mlflow_fit, mlflow_apply

def flatten_config(cfg):
    import pandas as pd
    return (
        pd.json_normalize(OmegaConf.to_object(cfg), sep="/")
        .to_dict(orient="records")[0]
    )

def _train_or_test(mode, model, data, trainer=None, seed=42,
                   mlflow=None, flat_conf={}, **_):

    pl.seed_everything(seed)

    model = instantiate(model)
    data = instantiate(data)
    trainer = instantiate(trainer)

    if mode == "train":
        if mlflow is not None:
            mlflow_fit(mlflow, trainer, model, data, flat_conf)
        else:
            trainer.fit(model, data)


    elif mode == "apply":
        if mlflow is not None:
            mlflow_apply(mlflow, trainer, model, data)
        else:
            raise NotImplementedError("Cannot `serotiny apply` without "
                                      "an MLFlow config.")
    else:
        raise ValueError(
            f"`mode` must be either 'train' or 'test'. Got '{mode}'"
        )

def train(cfg):
    if isinstance(cfg, dict):
        cfg = OmegaConf.create(cfg)

    _train_or_test("train", **cfg, flat_conf=flatten_config(cfg))

def apply(cfg):
    if isinstance(cfg, dict):
        cfg = OmegaConf.create(cfg)

    _train_or_test("apply", **cfg)
