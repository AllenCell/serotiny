import os
import logging
import inspect

from typing import List, Dict, Optional
from datetime import datetime

import fire
import pytorch_lightning as pl

import serotiny.datamodules as datamodules
import serotiny.models as models
from serotiny.models.zoo import get_checkpoint_callback, store_metadata
from serotiny.utils import module_get, module_or_path, get_classes_from_config, PATH_KEY, invoke_class

log = logging.getLogger(__name__)

def _get_kwargs():
    frame = inspect.currentframe().f_back
    keys, _, _, values = inspect.getargvalues(frame)
    kwargs = {}
    for key in keys:
        if key != 'self':
            kwargs[key] = values[key]
    return kwargs

def train_model(
    # model_name: str,
    model_config: Dict,
    # datamodule_name: str,
    datamodule_config: Dict,
    trainer_config: Dict,
    model_zoo_config: Dict,
    loggers_config: List[Dict] = [],
    callbacks_config: List[Dict] = [],
    gpu_ids: List[int] = [0],
    version_string: str = 'zero',
    seed: int = 42,
    metadata: Dict = {},
):
    called_args = _get_kwargs()

    pl.seed_everything(seed)

    model_zoo_path = model_zoo_config.get("path")
    checkpoint_config = model_zoo_config.get("checkpoint", {})

    model_name = model_config.get(PATH_KEY, 'UNDEFINED_MODEL_NAME')
    datamodule_name = datamodule_config.get(PATH_KEY, 'UNDEFINED_DATAMODULE_NAME')
    store_metadata(called_args, model_name, version_string, model_zoo_path)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(id) for id in gpu_ids])
    num_gpus = len(gpu_ids)
    num_gpus = (num_gpus if num_gpus != 0 else None)

    model = invoke_class(model_config)

    if version_string is None:
        version_string = "version_" + datetime.now().strftime("%d-%m-%Y--%H-%M-%S")

    log.info(f"creating datamodule {datamodule_name} with {datamodule_config}")

    datamodule = invoke_class(datamodule_config)
    datamodule.setup()

    loggers = path_invocations(loggers_config)

    if checkpoint_config:
        model_path = build_model_path(
            model_zoo_path,
            (model_name, version_string))
        config = {
            'dirpath': model_path,
            'filename': "epoch-{epoch:02d}"}
        config.update(checkpoint_config)
        checkpoint_callback = ModelCheckpoint(**config)
    else:
        checkpoint_callback = None

    if checkpoint_callback:
        trainer_config['checkpoint_callback'] = checkpoint_callback

    callbacks = path_invocations(callbacks_config)
    callbacks.append(checkpoint_callback)

    trainer = pl.Trainer(
        **trainer_config,
        logger=loggers,
        gpus=num_gpus,
        callbacks=callbacks,
    )

    log.info("Calling trainer.fit")
    trainer.fit(model, datamodule)

    trainer.test(datamodule=datamodule)


if __name__ == "__main__":
    fire.Fire(train_model)
