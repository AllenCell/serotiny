import logging

from typing import List, Dict
from datetime import datetime

import fire

log = logging.getLogger(__name__)


def _get_kwargs():
    import inspect
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
    test: bool = True,
    dynamic_imports_recurrent: bool = True,
    dynamic_imports_loaded_ok: bool = False,
):
    """
    Train a model given its configuration.

    Parameters
    ----------
    model: Dict
        The model configuration

    datamodule: Dict
        The datamodule configuration

    trainer: Dict
        The Pytorch Lightning Trainer configuration

    model_zoo: Dict
        The model zoo configuration, specifying how models shall be stored

    loggers: List[Dict]
        A list with the configuration of each logger to use

    callbacks: List[Dict]
        A list with the configuration of each callback to use

    gpu_ids: List[int]
        List of GPU ids to use

    version_string: str
        A string to tag the model and results with

    seed: int = 42
        Random seed

    test: bool = True
        Whether to run `trainer.test`

    dynamic_imports_recurrent: bool = True
        Whether to recursively apply the dynamic import functions

    dynamic_imports_loaded_ok: bool = False
        Whether to return the object passed into dynamic import functions,
        in case it's not loadable (e.g. has been loaded before this function
        was called)
    """

    # imports here to optimize CLI / Fire usage
    import pytorch_lightning as pl
    from serotiny.models.zoo import store_metadata, get_checkpoint_callback
    from serotiny.utils import INIT_KEY, load_config, load_multiple

    called_args = _get_kwargs()

    pl.seed_everything(seed, workers=True)

    model_config = model
    datamodule_config = datamodule
    trainer_config = trainer
    model_zoo_config = model_zoo
    loggers_config = loggers
    callbacks_config = callbacks

    model_zoo_path = model_zoo_config.get("path")
    model_class = model_config.get(INIT_KEY, "UNDEFINED_MODEL_CLASS")
    datamodule_name = datamodule_config.get(INIT_KEY, "UNDEFINED_DATAMODULE_NAME")

    store_metadata(called_args, model_class, version_string, model_zoo_path)

    model = load_config(model_config,
                        recurrent=dynamic_imports_recurrent,
                        loaded_ok=dynamic_imports_loaded_ok)

    if version_string is None:
        version_string = "version_" + datetime.now().strftime("%d-%m-%Y--%H-%M-%S")

    log.info(f"creating datamodule {datamodule_name} with {datamodule_config}")

    datamodule = load_config(datamodule_config,
                             recurrent=dynamic_imports_recurrent,
                             loaded_ok=dynamic_imports_loaded_ok)
    datamodule.setup()

    loggers = load_multiple(loggers_config,
                            recurrent=dynamic_imports_recurrent,
                            loaded_ok=dynamic_imports_loaded_ok)

    if len(model_zoo) > 0:
        config = {"filename": "epoch-{epoch:02d}"}
        checkpoint_config = model_zoo_config.get("checkpoint", {})
        config.update(checkpoint_config)

        checkpoint_callback = get_checkpoint_callback(
            model_class, version_string, model_zoo_path, **config
        )
    else:
        checkpoint_callback = None

    callbacks = load_multiple(callbacks_config,
                              recurrent=dynamic_imports_recurrent,
                              loaded_ok=dynamic_imports_loaded_ok)

    if checkpoint_callback is not None:
        callbacks.append(checkpoint_callback)

    if len(gpu_ids) == 0:
        gpu_ids = None

    trainer = pl.Trainer(
        **trainer_config,
        logger=loggers,
        gpus=gpu_ids,
        callbacks=callbacks,
    )

    log.info("Calling trainer.fit")
    trainer.fit(model, datamodule)

    if test:
        trainer.test(datamodule=datamodule, ckpt_path=None)


if __name__ == "__main__":
    fire.Fire(train_model)
