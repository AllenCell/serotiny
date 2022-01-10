import os
import logging
import inspect

from typing import List, Dict
from datetime import datetime

import fire
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from serotiny.models.zoo import store_metadata, build_model_path
from serotiny.utils import INIT_KEY, init_or_invoke, load_multiple

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
    model: Dict,
    datamodule: Dict,
    trainer: Dict,
    model_zoo: Dict,
    loggers: List[Dict] = [],
    callbacks: List[Dict] = [],
    gpu_ids: List[int] = [0],
    version_string: str = "zero",
    seed: int = 42,
    metadata: Dict = {},
):
    called_args = _get_kwargs()

    pl.seed_everything(seed)

    model_config = model
    datamodule_config = datamodule
    trainer_config = trainer
    model_zoo_config = model_zoo
    loggers_config = loggers
    callbacks_config = callbacks

    model_zoo_path = model_zoo_config.get("path")
    model_name = model_config.get(INIT_KEY, "UNDEFINED_MODEL_NAME")
    datamodule_name = datamodule_config.get(INIT_KEY, "UNDEFINED_DATAMODULE_NAME")

    store_metadata(called_args, model_name, version_string, model_zoo_path)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(id) for id in gpu_ids])
    num_gpus = len(gpu_ids)
    num_gpus = num_gpus if num_gpus != 0 else None

    model = init_or_invoke(model_config)

    if version_string is None:
        version_string = "version_" + datetime.now().strftime("%d-%m-%Y--%H-%M-%S")

    log.info(f"creating datamodule {datamodule_name} with {datamodule_config}")

    datamodule = init_or_invoke(datamodule_config)
    datamodule.setup()

    loggers = load_multiple(loggers_config)

    if len(model_zoo) > 0:
        model_path = build_model_path(model_zoo_path, (model_name, version_string))
        config = {"dirpath": model_path, "filename": "{epoch:02d}"}
        checkpoint_config = model_zoo_config.get("checkpoint", {})
        config.update(checkpoint_config)
        checkpoint_callback = ModelCheckpoint(**config)
    else:
        checkpoint_callback = None

    callbacks = load_multiple(callbacks_config)
    if checkpoint_callback is not None:
        callbacks.append(checkpoint_callback)

    trainer = pl.Trainer(
        **trainer_config, logger=loggers, gpus=num_gpus, callbacks=callbacks,
    )

    log.info("Calling trainer.fit")
    trainer.fit(model, datamodule)
    trainer.test(datamodule=datamodule, ckpt_path=None)


if __name__ == "__main__":
    fire.Fire(train_model)
