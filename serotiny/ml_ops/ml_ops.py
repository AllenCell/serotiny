import logging
import warnings
import sys

from omegaconf import OmegaConf
from hydra.utils import get_original_cwd


# silence aicsimageio related warnings
warnings.filterwarnings(action="ignore", category=FutureWarning, module="ome_types")
logging.getLogger("xmlschema").setLevel(logging.WARNING)
logging.getLogger("bfio").setLevel(logging.WARNING)
logging.getLogger("bfio.backends").setLevel(logging.WARNING)
logging.getLogger("ome_zarr").setLevel(logging.WARNING)
logging.getLogger("ome_zarr.reader").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def instantiate(cfg):
    from hydra.utils import instantiate as _instantiate
    from copy import copy

    if not isinstance(cfg, (list, dict)):
        _cfg = OmegaConf.to_container(cfg, resolve=True)
    else:
        _cfg = copy(cfg)

    if "_aux_" in _cfg:
        del _cfg["_aux_"]

    return _instantiate(_cfg)


def _do_model_op(
    mode,
    model,
    data,
    trainer=None,
    seed=42,
    mlflow=None,
    full_conf={},
    test=False,
    predict=False,
    tune=False,
    multiprocessing_strategy=None,
    make_notebook=None,
    **_,
):

    from pathlib import Path

    import pytorch_lightning as pl

    from .mlflow_utils import mlflow_fit, mlflow_predict, mlflow_test
    from .utils import make_notebook as mk_notebook
    from .utils import save_model_predictions

    if multiprocessing_strategy is not None:
        import torch

        torch.multiprocessing.set_sharing_strategy(multiprocessing_strategy)

    if mode in ["train", "test", "predict"]:
        if make_notebook is not None:
            mk_notebook(full_conf, make_notebook)
            return

        pl.seed_everything(seed)
        logger.info("Instantiating datamodule")
        data = instantiate(data)
        logger.info("Instantiating trainer")
        trainer = instantiate(trainer)
        logger.info("Instantiating model")
        model = instantiate(model)

        if mode == "train":
            if mlflow is not None and mlflow.get("tracking_uri") is not None:
                mlflow_fit(
                    mlflow,
                    trainer,
                    model,
                    data,
                    full_conf=full_conf,
                    test=test,
                    predict=predict,
                    tune=tune,
                )
            else:
                if tune:
                    logger.info("Calling trainer.tune")
                    trainer.tune(model, data)

                logger.info("Calling trainer.fit")
                trainer.fit(model, data)

        elif mode == "test":
            if mlflow is not None and mlflow.get("tracking_uri") is not None:
                mlflow_test(mlflow, trainer, data, full_conf=full_conf)
            else:
                if "ckpt_path" not in full_conf:
                    raise NotImplementedError(
                        "Cannot `serotiny test` without "
                        "an MLFlow config, or a local ckpt_path."
                    )
                ckpt_path = full_conf["ckpt_path"]
                trainer.test(model, data, ckpt_path=ckpt_path)

        elif mode == "predict":
            if mlflow is not None and mlflow.get("tracking_uri") is not None:
                mlflow_predict(mlflow, trainer, data, full_conf=full_conf)
            else:
                if "ckpt_path" not in full_conf:
                    raise NotImplementedError(
                        "Cannot `serotiny predict` without "
                        "an MLFlow config, or a local ckpt_path."
                    )

                preds_dir_default = Path(get_original_cwd()) / "predictions"
                preds_dir = Path(
                    full_conf.get("predictions_output_dir", preds_dir_default)
                )
                preds_dir.mkdir(exist_ok=True, parents=True)

                ckpt_path = full_conf["ckpt_path"]
                preds = trainer.predict(model, data, ckpt_path=ckpt_path)
                save_model_predictions(model, preds, preds_dir)

    else:
        raise ValueError(f"`mode` must be 'train', 'test' or 'predict'. Got '{mode}'")


def _do_model_op_wrapper(cfg):
    if isinstance(cfg, dict):
        cfg = OmegaConf.create(cfg)

    # there might be other dots in the
    # executable path, ours is the last
    mode = sys.argv[0].split(".")[-1]
    if mode in ["train", "predict", "test"]:
        cfg = OmegaConf.merge(cfg, {"mode": mode})

    _do_model_op(
        **cfg,
        full_conf=cfg,
    )
