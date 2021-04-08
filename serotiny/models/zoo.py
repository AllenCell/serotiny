from pathlib import Path
import os

import serotiny.models as models

def _get_root(model_root=None):
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
    model_root = _get_root(model_root)

    if not model_root.exists():
        raise FileNotFoundError("Given model_root does not exists.")

    model_class, model_id = model_path.split("/")
    model_class = models.__dict__[model_class]

    model_id = (model_id if ".ckpt" in model_id else model_id+".ckpt")
    model_path = (model_root / model_class) / model_id

    return model_class.load_from_checkpoint(checkpoint_path=model_path)

def get_trainer_at_checkpoint(model_path, model_root=None):
    model_root = _get_root(model_root)

    if not model_root.exists():
        raise FileNotFoundError("Given model_root does not exists.")

    model_class, model_id = model_path.split("/")
    model_class = models.__dict__[model_class]

    model_id = (model_id if ".ckpt" in model_id else model_id+".ckpt")
    model_path = (model_root / model_class) / model_id

    trainer = Trainer(resume_from_checkpoint=model_path)


def store_model(trainer, model_class, model_id, model_root=None):
    model_root = _get_root(model_root)

    if not model_root.exists():
        model_root.mkdir(parents=True)

    model_path = (model_root / model_class)
    if not model_path.exists():
        model_path.mkdir(parents=True)

    model_id = (model_id if ".ckpt" in model_id else model_id+".ckpt")

    model_path = model_path / model_id
    trainer.save_checkpoint(model_path)
