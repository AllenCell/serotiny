from typing import Optional, Union, Dict
from pathlib import Path
import os

import omegaconf
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from serotiny import models
from serotiny.utils import load_multiple, module_or_path


def get_root(zoo_root: Optional[Union[str, Path]] = None):
    """
    Retrieve the model zoo root. If a path is specified, that is
    the used model zoo root. Otherwise, the environment variable
    SEROTINY_ZOO_ROOT is checked. If it doesn't exist and the
    user has a .cache folder in their folder, the model zoo root
    is set to ~/.cache/serotiny/zoo

    Parameters
    ----------
    zoo_root: Optional[Union[str, Path]]
        Path to override the model zoo root
    """
    if zoo_root is not None:
        zoo_root = Path(zoo_root)
    elif "SEROTINY_ZOO_ROOT" in os.environ:
        zoo_root = Path(os.environ["SEROTINY_ZOO_ROOT"])
    elif Path("~/.cache").expanduser().exists():
        zoo_root = Path("~/.cache").expanduser() / "serotiny/zoo"
    else:
        raise ValueError("No valid model zoo root")

    return zoo_root


def _get_checkpoint(
    model_class: str,
    version_string: str,
    zoo_root: Optional[Union[str, Path]] = None,
):
    zoo_root = get_root(zoo_root)

    if not zoo_root.exists():
        raise FileNotFoundError("Given zoo_root does not exists.")

    version_string = version_string.replace("/", "_")

    model_path = (zoo_root / model_class) / version_string

    with open(str(model_path) + ".yaml") as f:
        config = yaml.load(f, Loader=yaml.UnsafeLoader)

    ckpt_path = list(model_path.glob("*.ckpt"))[0]

    return ckpt_path, config


def get_model(
    model_class: str,
    version_string: str,
    zoo_root: Optional[Union[str, Path]] = None,
):
    """
    Retrieve a model stored in disk, inside a model zoo.

    Parameters
    ----------
    model_class: str
        The "import path" to the model class, e.g.
        serotiny.models.RegressionModel

    version_string:
        a version string that uniquely identifies the model within the
        zoo.

    zoo_root: Optional[Union[str, Path]] = None
        (Optional) path to the model zoo root
    """

    ckpt_path, config = _get_checkpoint(model_class, version_string, zoo_root)
    model_class = module_or_path(models, model_class)

    model_config = config["model"]

    return model_class.load_from_checkpoint(checkpoint_path=ckpt_path, **model_config)


def store_model(trainer, model_class, version_string, zoo_root=None):
    """
    Stored a model in disk, inside a model zoo.

    Parameters
    ----------
    trainer: Trainer
        A Pytorch Lightning Trainer instance which has trained the model

    model_class: str
        A string containing the "import path" to the model class,
        e.g. serotiny.models.RegressionModel

    version_string: str
        A version string that uniquely identifies the model within the
        zoo.

    zoo_root: Optional[Union[str, Path]] = None
        (Optional) path to the model zoo root
    """
    zoo_root = get_root(zoo_root)

    if not zoo_root.exists():
        zoo_root.mkdir(parents=True, exist_ok=True)

    model_path = zoo_root / model_class
    if not model_path.exists():
        model_path.mkdir(parents=True, exist_ok=True)

    version_string = version_string if ".ckpt" in version_string else version_string + ".ckpt"
    version_string = version_string.replace("/", "_")

    model_path = model_path / version_string
    trainer.save_checkpoint(model_path)


def get_checkpoint_callback(
    model_class: str,
    version_string: str,
    zoo_root: Optional[Union[str, Path]] = None,
    **checkpoint_callback_kwargs
):
    """
    Get an instantiated ModelCheckpoint callback, configured to use
    a specified model zoo

    Parameters
    ----------
    model_class: str
        A string containing the "import path" to the model class,
        e.g. serotiny.models.RegressionModel

    version_string: str
        A version string that uniquely identifies the model within the
        zoo.

    zoo_root: Optional[Union[str, Path]] = None
        (Optional) path to the model zoo root

    checkpoint_callback_kwargs: keyword arguments
        Additional arguments to ModelCheckpoint. See Pytorch Lightning docs
        here: https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.model_checkpoint.html

    """

    zoo_root = get_root(zoo_root)
    version_string = version_string.replace("/", "_")
    model_path = (zoo_root / model_class) / version_string
    model_path.mkdir(parents=True, exist_ok=True)

    if "filename" not in checkpoint_callback_kwargs:
        checkpoint_callback_kwargs["filename"] = "{epoch:02d}"

    return ModelCheckpoint(
        dirpath=model_path, **checkpoint_callback_kwargs
    )


def get_trainer_at_checkpoint(
    model_class: str,
    version_string: str,
    zoo_root: Optional[Union[str, Path]] = None,
    reload_callbacks: bool = False,
    reload_loggers: bool = True,
):
    """
    Retrieve a pytorch lightning trainer's state stored in disk,
    inside a model zoo.

    Parameters
    ----------
    model_class: str
        A string containing the "import path" to the model class,
        e.g. serotiny.models.RegressionModel

    version_string: str
        A version string that uniquely identifies the model within the
        zoo.

    zoo_root: Optional[Union[str, Path]] = None
        (Optional) path to the model zoo root

    reload_callbacks: bool = False
        Flag to determine whether to reload the trainer's callbacks

    reload_loggers: bool = False
        Flag to determine whether to reload the trainer's loggers
    """

    ckpt_path, config = _get_checkpoint(model_class, version_string, zoo_root)

    trainer_config = config["trainer"]

    model_zoo_config = config["model_zoo"]
    checkpoint_callback = get_checkpoint_callback(
        model_class,
        version_string,
        model_zoo_config.get("checkpoint_monitor"),
        model_zoo_config.get("checkpoint_mode", "min"),
        zoo_root,
    )

    checkpoint_callback.best_model_path = str(ckpt_path)
    loggers = load_multiple(config["loggers"]) if reload_loggers else None

    callbacks = [checkpoint_callback]
    if reload_callbacks:
        callbacks += load_multiple(config["callbacks"])

    trainer = Trainer(
        resume_from_checkpoint=ckpt_path,
        **trainer_config,
        callbacks=callbacks,
        logger=loggers
    )

    return trainer


def store_metadata(metadata: Dict, model_class: str, version_string: str, zoo_root=None):
    """
    Store metadata associated with a training run of a model

    Parameters
    ----------
    metadata: Dict
        Metadata to be stored in a yaml

    model_class: str
        The "import path" to the model class, e.g.
        serotiny.models.RegressionModel

    version_string:
        a version string that uniquely identifies the model within the
        zoo.

    zoo_root: Optional[Union[str, Path]] = None
        (Optional) path to the model zoo root

    """

    version_string = version_string.replace("/", "_")
    zoo_root = get_root(zoo_root)

    if not zoo_root.exists():
        zoo_root.mkdir(parents=True, exist_ok=True)

    model_path = zoo_root / model_class
    if not model_path.exists():
        model_path.mkdir(parents=True, exist_ok=True)

    version_string = version_string if ".yaml" in version_string else version_string + ".yaml"

    model_path = model_path / version_string

    if isinstance(metadata, omegaconf.DictConfig):
        omegaconf.OmegaConf.save(metadata, model_path, resolve=True)
    else:
        with open(model_path, "w") as f:
            yaml.dump(metadata, f)
