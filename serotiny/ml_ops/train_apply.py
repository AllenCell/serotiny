import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import OmegaConf
import copy

from .mlflow_utils import mlflow_fit, mlflow_apply

def flatten_config(cfg):
    import pandas as pd
    conf = (
        pd.json_normalize(cfg, sep="/")
        .to_dict(orient="records")[0]
    )
    keys = list(conf.keys())

    for k in keys:
        try:
            sub_conf = flatten_config(conf[k])
            conf.update({f"{k}/{_k}" for k,v in sub_conf.items()})
            del conf[k]
            continue
        except:
            pass

        if isinstance(conf[k], list):
            for i, el in enumerate(conf[k]):
                try:
                    sub_conf = flatten_config(el)
                    conf.update({f"{k}/{_k}" for k,v in sub_conf.items()})
                except Exception as e:
                    conf[f"{k}/{i}"] = el
            del conf[k]

    return (
        pd.json_normalize(conf, sep="/")
        .to_dict(orient="records")[0]
    )

def _train_or_test(mode, model, data, trainer=None, seed=42,
                   mlflow=None, flat_conf={}, test=False,
                   multiprocessing_strategy=None, **_):

    pl.seed_everything(seed)

    if multiprocessing_strategy is not None:
        import torch
        torch.multiprocessing.set_sharing_strategy(multiprocessing_strategy)

    model = instantiate(model)
    data = instantiate(data)
    trainer = instantiate(trainer)

    if mode == "train":
        if mlflow is not None:
            mlflow_fit(mlflow, trainer, model, data, flat_conf, test)
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

    flat_conf = flatten_config(OmegaConf.to_object(cfg))
    _train_or_test("train", **cfg, flat_conf=flat_conf)

def apply(cfg):
    if isinstance(cfg, dict):
        cfg = OmegaConf.create(cfg)

    _train_or_test("apply", **cfg)
