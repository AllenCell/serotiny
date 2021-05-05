from pathlib import Path
import os
import omegaconf


from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import serotiny.models as models


def get_root(model_root=None):
    if model_root is not None:
        model_root = Path(model_root)
    elif "SEROTINY_ZOO_ROOT" in os.environ:
        model_root = Path(os.environ["SEROTINY_ZOO_ROOT"])
    elif Path("~/.cache").expanduser().exists():
        model_root = Path("~/.cache").expanduser() / "serotiny/zoo"
    else:
        raise ValueError("No valid model zoo root")

    return model_root


def get_model(model_path, model_root=None):
    model_root = get_root(model_root)

    if not model_root.exists():
        raise FileNotFoundError("Given model_root does not exists.")

    model_class, model_id = model_path.split("/")
    model_class = models.__dict__[model_class]

    model_id = model_id if ".ckpt" in model_id else model_id + ".ckpt"
    model_path = (model_root / model_class) / model_id

    return model_class.load_from_checkpoint(checkpoint_path=model_path)


def get_trainer_at_checkpoint(model_path, model_root=None):
    model_root = get_root(model_root)

    if not model_root.exists():
        raise FileNotFoundError("Given model_root does not exists.")

    model_class, model_id = model_path.split("/")
    model_class = models.__dict__[model_class]

    model_id = model_id if ".ckpt" in model_id else model_id + ".ckpt"
    model_path = (model_root / model_class) / model_id

    return Trainer(resume_from_checkpoint=model_path)


def store_model(trainer, model_class, model_id, model_root=None):
    model_root = get_root(model_root)

    if not model_root.exists():
        model_root.mkdir(parents=True)

    model_path = model_root / model_class
    if not model_path.exists():
        model_path.mkdir(parents=True)

    model_id = model_id if ".ckpt" in model_id else model_id + ".ckpt"

    model_path = model_path / model_id
    trainer.save_checkpoint(model_path)

def get_checkpoint_callback(model_class, model_id, checkpoint_monitor,
                            checkpoint_mode, model_root=None):
    model_root = get_root(model_root)

    if not model_root.exists():
        model_root.mkdir(parents=True)

    model_path = model_root / model_class
    if not model_path.exists():
        model_path.mkdir(parents=True)

    model_id = model_id.split(".ckpt")[0]

    model_path = model_path / model_id
    if not model_path.exists():
        model_path.mkdir(parents=True)

    return ModelCheckpoint(
        monitor=checkpoint_monitor,
        mode=checkpoint_mode,
        dirpath=model_path,
        filename="epoch{epoch:02d}"
    )

def store_called_args(called_args, model_class, model_id, model_root=None):
    model_root = get_root(model_root)

    if not model_root.exists():
        model_root.mkdir(parents=True)

    model_path = model_root / model_class
    if not model_path.exists():
        model_path.mkdir(parents=True)

    model_id = model_id if ".yaml" in model_id else model_id + ".yaml"

    model_path = model_path / model_id

    omegaconf.OmegaConf.save(called_args, model_path, resolve=True)
