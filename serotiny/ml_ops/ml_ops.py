import logging
import sys

logger = logging.getLogger(__name__)


def _do_model_op(
    mode,
    model,
    data,
    trainer=None,
    seed=42,
    mlflow=None,
    full_conf={},
    test=False,
    multiprocessing_strategy=None,
    make_notebook=None,
    **_,
):

    from pathlib import Path

    import pytorch_lightning as pl
    from hydra.utils import instantiate

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
                raise NotImplementedError(
                    "Cannot `serotiny test` without " "an MLFlow config."
                )

        elif mode == "predict":
            if mlflow is not None and mlflow.get("tracking_uri") is not None:
                mlflow_predict(mlflow, trainer, model, data, full_conf=full_conf)
            else:
                if "ckpt_path" not in full_conf:
                    raise NotImplementedError(
                        "Cannot `serotiny predict` without "
                        "an MLFlow config, or a local ckpt_path."
                    )
                preds_dir = Path(
                    full_conf.get("predictions_output_dir", "./predictions")
                )
                preds_dir.mkdir(exist_ok=True, parents=True)

                ckpt_path = full_conf["ckpt_path"]
                preds = trainer.predict(model, data, ckpt_path=ckpt_path)
                save_model_predictions(model, preds, preds_dir)

    else:
        raise ValueError(f"`mode` must be 'train', 'test' or 'predict'. Got '{mode}'")


def _do_model_op_wrapper(cfg):
    if isinstance(cfg, dict):
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(cfg)

    _do_model_op(
        sys.argv[0].split(".")[-1],  # there might be other dots in the
        # executable path, ours is the last
        **cfg,
        full_conf=cfg,
    )