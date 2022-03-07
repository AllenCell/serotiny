import logging
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import OmegaConf

from .mlflow_utils import mlflow_fit, mlflow_test

logger = logging.getLogger(__name__)

def _do_model_op(mode, model, data, trainer=None, seed=42,
                 mlflow=None, full_conf={}, test=False,
                 multiprocessing_strategy=None, **_):

    pl.seed_everything(seed)

    if multiprocessing_strategy is not None:
        import torch
        torch.multiprocessing.set_sharing_strategy(multiprocessing_strategy)

    logger.info("Instantiating model, datamodule and trainer")
    model = instantiate(model)
    data = instantiate(data)
    trainer = instantiate(trainer)

    if mode == "train":
        if mlflow is not None:
            mlflow_fit(mlflow, trainer, model, data, full_conf=full_conf, test=test)
        else:
            logger.info("Calling trainer.fit")
            trainer.fit(model, data)

    elif mode == "test":
        if mlflow is not None:
            mlflow_test(mlflow, trainer, model, data, full_conf=full_conf)
        else:
            raise NotImplementedError("Cannot `serotiny test` without "
                                      "an MLFlow config.")

    elif mode == "predict":
        if mlflow is not None:
            #mlflow_predict(mlflow, trainer, model, data, full_conf=full_conf)
            raise NotImplementedError
        else:
            raise NotImplementedError

    else:
        raise ValueError(
            f"`mode` must be 'train', 'test' or 'predict'. Got '{mode}'"
        )


def train(cfg):
    if isinstance(cfg, dict):
        cfg = OmegaConf.create(cfg)

    _do_model_op("train", **cfg, full_conf=cfg)


def test(cfg):
    if isinstance(cfg, dict):
        cfg = OmegaConf.create(cfg)

    _do_model_op("test", **cfg, full_conf=cfg)


def predict(cfg):
    if isinstance(cfg, dict):
        cfg = OmegaConf.create(cfg)

    _do_model_op("predict", **cfg, full_conf=cfg)
