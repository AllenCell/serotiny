import os
import logging
import inspect
import yaml
from typing import List, Dict
from datetime import datetime

import fire
import pytorch_lightning as pl

from serotiny.models.zoo import store_metadata, get_checkpoint_callback
from serotiny.utils import INIT_KEY, load_config, load_multiple

log = logging.getLogger(__name__)


def _get_kwargs():
    frame = inspect.currentframe().f_back
    keys, _, _, values = inspect.getargvalues(frame)
    kwargs = {}
    for key in keys:
        if key != "self":
            kwargs[key] = values[key]
    return kwargs


def train_model(
    model: Dict = None,
    datamodule: Dict = None,
    trainer: Dict = None,
    model_zoo: Dict = None,
    loggers: List[Dict] = None,
    callbacks: List[Dict] = None,
    gpu_ids: List[int] = None,
    version_string: str = None,
    seed: int = None,
    config: Dict = None,
    configfile: str = None,
):
    called_args = _get_kwargs()

    if configfile:
        with open(configfile, "r") as yaml_cf:
            config = yaml.load(yaml_cf)

    if config:
        model_config = config.get("model", {})
        datamodule_config = config.get("datamodule", {})
        trainer_config = config.get("trainer", {})
        model_zoo_config = config.get("model_zoo", {})
        loggers_config = config.get("loggers", [])
        callbacks_config = config.get("callbacks", [])
        gpu_ids = config.get("gpu_ids", [0])
        version_string = config.get("version_string", "zero")
        seed = config.get("seed", 42)
    else:
        model_config = model or {}
        datamodule_config = datamodule or {}
        trainer_config = trainer or {}
        model_zoo_config = model_zoo or {}
        loggers_config = loggers or []
        callbacks_config = callbacks or []
        gpu_ids = gpu_ids or [0]
        version_string = version_string or "zero"
        seed = seed or 42

    pl.seed_everything(seed)

    model_zoo_path = model_zoo_config.get("path")
    model_name = model_config.get(INIT_KEY, "UNDEFINED_MODEL_NAME")
    datamodule_name = datamodule_config.get(INIT_KEY, "UNDEFINED_DATAMODULE_NAME")

    store_metadata(called_args, model_name, version_string, model_zoo_path)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(id) for id in gpu_ids])
    num_gpus = len(gpu_ids)
    num_gpus = num_gpus if num_gpus != 0 else None

    model = load_config(model_config)

    if version_string is None:
        version_string = "version_" + datetime.now().strftime("%d-%m-%Y--%H-%M-%S")

    log.info(f"creating datamodule {datamodule_name} with {datamodule_config}")

    datamodule = load_config(datamodule_config)
    datamodule.setup()

    loggers = load_multiple(loggers_config)

    if len(model_zoo_config) > 0:
        zoo_config = {"filename": "epoch-{epoch:02d}"}
        checkpoint_config = model_zoo_config.get("checkpoint", {})
        zoo_config.update(checkpoint_config)

        checkpoint_callback = get_checkpoint_callback(
            model_name, version_string, model_zoo_path, **zoo_config
        )
    else:
        checkpoint_callback = None

    callbacks = load_multiple(callbacks_config)
    if checkpoint_callback is not None:
        callbacks.append(checkpoint_callback)

    trainer = pl.Trainer(
        **trainer_config,
        logger=loggers,
        gpus=num_gpus,
        callbacks=callbacks,
    )

    log.info("Calling trainer.fit")
    trainer.fit(model, datamodule)
    trainer.test(datamodule=datamodule, ckpt_path=None)


if __name__ == "__main__":
    fire.Fire(train_model)
