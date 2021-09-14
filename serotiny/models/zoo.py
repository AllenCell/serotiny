from pathlib import Path
import os
import omegaconf
import yaml

from collections import OrderedDict
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import serotiny.models as models
from serotiny.utils import load_multiple, module_or_path


def ordered_dump(data, stream=None, Dumper=yaml.SafeDumper, **kwds):
    class OrderedDumper(Dumper):
        pass
    def _dict_representer(dumper, data):
        return dumper.represent_mapping(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            data.items())
    OrderedDumper.add_representer(OrderedDict, _dict_representer)
    return yaml.dump(data, stream, OrderedDumper, **kwds)


def ordered_load(stream, Loader=yaml.SafeLoader, object_pairs_hook=OrderedDict):
    class OrderedLoader(Loader):
        pass
    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))
    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)
    return yaml.load(stream, OrderedLoader)


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

def _get_checkpoint(model_path, model_root):
    model_root = get_root(model_root)

    if not model_root.exists():
        raise FileNotFoundError("Given model_root does not exists.")

    _split = model_path.split("/")
    model_class_name = _split[0]
    model_id = "_".join(_split[1:])

    model_path = (model_root / model_class_name) / model_id

    with open(str(model_path) + ".yaml") as f:
        config = yaml.load(f, Loader=yaml.UnsafeLoader)

    ckpt_path = list(model_path.glob("*.ckpt"))[0]

    return ckpt_path, model_class_name, config


def get_model(model_path, model_root=None):
    ckpt_path, model_class_name, config = _get_checkpoint(model_path, model_root)
    model_class = module_or_path(models, model_class_name)

    model_config = config["model"]

    return model_class.load_from_checkpoint(checkpoint_path=ckpt_path, **model_config)


def get_trainer_at_checkpoint(
        model_path,
        model_root=None,
        reload_callbacks=False,
        reload_loggers=True):

    ckpt_path, model_class_name, config = _get_checkpoint(model_path, model_root)

    trainer_config = config["trainer_config"]

    model_zoo_config = config["model_zoo_config"]
    checkpoint_callback = get_checkpoint_callback(
        model_class_name,
        model_path.split("/")[1].split('.ckpt')[0],
        model_zoo_config.get("checkpoint_monitor"),
        model_zoo_config.get("checkpoint_mode"),
        model_root,
    )

    checkpoint_callback.best_model_path = str(ckpt_path)
    loggers = (load_multiple(config["loggers"])
               if reload_loggers else None)

    callbacks = [checkpoint_callback]
    if reload_callbacks:
        callbacks += load_multiple(config["callbacks"])

    trainer = Trainer(resume_from_checkpoint=ckpt_path,
                      **trainer_config,
                      callbacks=callbacks,
                      logger=loggers)

    return trainer


def store_model(trainer, model_class, model_id, model_root=None):
    model_root = get_root(model_root)

    if not model_root.exists():
        model_root.mkdir(parents=True, exist_ok=True)

    model_path = model_root / model_class
    if not model_path.exists():
        model_path.mkdir(parents=True, exist_ok=True)

    model_id = model_id if ".ckpt" in model_id else model_id + ".ckpt"

    model_path = model_path / model_id
    trainer.save_checkpoint(model_path)


def build_model_path(model_root, path):
    model_path = get_root(model_root)

    if not model_path.exists():
        model_path.mkdir(parents=True, exist_ok=True)

    for step in path:
        model_path = model_path / step
        if not model_path.exists():
            model_path.mkdir(parents=True, exist_ok=True)

    return model_path


def get_checkpoint_callback(
        model_class,
        model_id,
        checkpoint_monitor,
        checkpoint_mode,
        model_root=None):

    model_path = build_model_path(model_root, (model_class, model_id))

    return ModelCheckpoint(
        monitor=checkpoint_monitor,
        mode=checkpoint_mode,
        dirpath=model_path,
        filename="epoch{epoch:02d}"
    )


def store_metadata(metadata, model_class, model_id, model_root=None):
    model_root = get_root(model_root)

    if not model_root.exists():
        model_root.mkdir(parents=True, exist_ok=True)

    model_path = model_root / model_class
    if not model_path.exists():
        model_path.mkdir(parents=True, exist_ok=True)

    model_id = model_id if ".yaml" in model_id else model_id + ".yaml"

    model_path = model_path / model_id.replace("/", "_")

    if isinstance(metadata, omegaconf.DictConfig):
        omegaconf.OmegaConf.save(metadata, model_path, resolve=True)
    else:
        with open(model_path, "w") as f:
            yaml.dump(metadata, f)
