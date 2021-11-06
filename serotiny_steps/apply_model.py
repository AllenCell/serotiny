import os

from typing import Dict, List

import fire

def apply_model(
    model_class: str,
    model_id: str,
    datamodule: Dict,
    trainer: Dict,
    model_zoo: Dict,
    callbacks: Dict,
    gpu_ids: List[int],
    loggers: List[Dict] = [],
    seed: int = 42,
):
    """
    Apply a train model to some data

    Parameters
    ----------
    model_class: str
        The "import path" to the model class, e.g. serotiny.models.RegressionModel

    model_id:
        A version string that uniquely identifies the model within the
        zoo.

    datamodule: Dict
        The datamodule configuration

    trainer: Dict
        The Pytorch Lightning Trainer configuration

    model_zoo: Dict
        The model zoo configuration, specifying where models shall be loaded from

    callbacks: List[Dict]
        A list with the configuration of each callback to use

    gpu_ids: List[int]
        List of GPU ids to use

    loggers: List[Dict]
        A list with the configuration of each logger to use

    seed:
        Random seed
    """


    # imports here to optimize CLI / Fire
    import pytorch_lightning as pl
    from serotiny.models.zoo import get_model
    from serotiny.utils import init, load_multiple

    pl.seed_everything(seed)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(id) for id in gpu_ids])
    num_gpus = len(gpu_ids)
    num_gpus = num_gpus if num_gpus != 0 else None

    datamodule_config = datamodule
    trainer_config = trainer
    model_zoo_config = model_zoo
    callbacks_config = callbacks
    loggers_config = loggers

    model_zoo_path = model_zoo_config.get("path")
    model = get_model(model_class, model_id, model_zoo_path)

    datamodule = init(datamodule_config)
    datamodule.setup()

    loggers = load_multiple(loggers_config)
    callbacks = load_multiple(callbacks_config)

    trainer = pl.Trainer(
        **trainer_config,
        logger=loggers,
        callbacks=callbacks,
        gpus=num_gpus,
    )

    trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    fire.Fire(apply_model)
