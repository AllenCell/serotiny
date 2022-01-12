from hydra.utils import instantiate
from pytorch_lightning import seed_everything

from .utils import add_mlflow_conf

def _train_or_test(mode, model, data, trainer=None, seed=42,
                   mlflow=None, **_):
    seed_everything(42)

    model = instantiate(model)
    data = instantiate(data)

    trainer = add_mlflow_conf(trainer, mlflow)
    trainer = instantiate(trainer)

    if mode == "train":
        trainer.fit(model, data)
    elif mode == "apply":
        trainer.test(model, data)
    else:
        raise ValueError(
            f"`mode` must be either 'train' or 'test'. Got '{mode}'"
        )

def train(cfg):
    _train_or_test("train", **cfg)

def apply(cfg):
    _train_or_test("apply", **cfg)
