import sys
import logging

logger = logging.getLogger(__name__)

def _do_model_op(mode, model, data, trainer=None, seed=42,
                 mlflow=None, full_conf={}, test=False,
                 multiprocessing_strategy=None, make_notebook=None,
                 **_):

    import pytorch_lightning as pl
    from hydra.utils import instantiate
    from .mlflow_utils import mlflow_fit, mlflow_test
    from .utils import make_notebook as mk_notebook

    if multiprocessing_strategy is not None:
        import torch
        torch.multiprocessing.set_sharing_strategy(multiprocessing_strategy)

    if mode in ["train", "test", "predict"]:
        if make_notebook is not None:
            mk_notebook(full_conf, make_notebook)
            return

        pl.seed_everything(seed)
        logger.info("Instantiating model, datamodule and trainer")
        model = instantiate(model)
        data = instantiate(data)
        trainer = instantiate(trainer)

        if mode == "train":
            if mlflow is not None and mlflow.get("tracking_uri") is not None:
                mlflow_fit(mlflow, trainer, model, data, full_conf=full_conf, test=test)
            else:
                logger.info("Calling trainer.fit")
                trainer.fit(model, data)

        elif mode == "test":
            if mlflow is not None and mlflow.get("tracking_uri") is not None:
                mlflow_test(mlflow, trainer, model, data, full_conf=full_conf)
            else:
                raise NotImplementedError("Cannot `serotiny test` without "
                                          "an MLFlow config.")

        elif mode == "predict":
            if mlflow is not None and mlflow.get("tracking_uri") is not None:
                #mlflow_predict(mlflow, trainer, model, data, full_conf=full_conf)
                raise NotImplementedError
            else:
                raise NotImplementedError

    else:
        raise ValueError(
            f"`mode` must be 'train', 'test' or 'predict'. Got '{mode}'"
        )


def _do_model_op_wrapper(cfg):
    if isinstance(cfg, dict):
        from omegaconf import OmegaConf
        cfg = OmegaConf.create(cfg)

    _do_model_op(sys.argv[0].split(".")[-1], # there might be other dots in the
                                             # executable path, ours is the last
                 **cfg,
                 full_conf=cfg)












