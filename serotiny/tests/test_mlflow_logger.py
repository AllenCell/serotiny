import os
from pathlib import Path
from unittest import mock

import pytest

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.demos.boring_classes import BoringModel
from pytorch_lightning.callbacks import ModelCheckpoint

from serotiny.loggers import MLFlowLogger
from serotiny.models.utils.mlflow import get_config


@mock.patch("pytorch_lightning.loggers.mlflow._MLFLOW_AVAILABLE", return_value=True)
@pytest.mark.parametrize("monitor", [True, False])
def test_mlflow_log_model(_, tmpdir, monitor):
    """Test that the logger creates the folders and files in the right
    place."""

    max_epochs = 10
    limit_train_batches = 3
    limit_val_batches = 3
    monitor_key = "val_log"

    class CustomBoringModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.train_log_epochs = torch.randn(max_epochs, limit_train_batches)
            self.val_logs = torch.randn(max_epochs, limit_val_batches)
            self.scores = []

        def training_step(self, batch, batch_idx):
            log_value = self.train_log_epochs[self.current_epoch, batch_idx]
            self.log("train_log", log_value, on_epoch=True)
            return super().training_step(batch, batch_idx)

        def validation_step(self, batch, batch_idx):
            log_value = self.val_logs[self.current_epoch, batch_idx]
            self.log(monitor_key, log_value)
            return super().validation_step(batch, batch_idx)

        def on_train_epoch_end(self):
            if "train" in monitor_key:
                self.scores.append(self.trainer.logged_metrics[monitor_key])

        def on_validation_epoch_end(self):
            if not self.trainer.sanity_checking and "val" in monitor_key:
                self.scores.append(self.trainer.logged_metrics[monitor_key])

    model = CustomBoringModel()

    mlflow_root = os.path.join(tmpdir, "mlflow/")
    local_ckpt_root = os.path.join(tmpdir, "local_ckpt")

    logger = MLFlowLogger("test", save_dir=mlflow_root)

    if monitor:
        checkpoint_callback = ModelCheckpoint(
            dirpath=local_ckpt_root, save_top_k=2, monitor=monitor_key
        )
    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath=local_ckpt_root, save_top_k=1, monitor=None
        )

    trainer = Trainer(
        default_root_dir=tmpdir,
        logger=logger,
        callbacks=[checkpoint_callback],
        max_epochs=max_epochs,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
    )
    trainer.fit(model)

    run_folders = [
        _ for _ in Path(mlflow_root).glob("*") if _.name not in ("0", ".trash")
    ]
    assert len(run_folders) == 1
    run_root = run_folders.pop()

    if monitor:
        ckpt_folder = os.path.join(
            run_root,
            f"{logger.run_id}/artifacts/checkpoints/{monitor_key}",
        )

        assert len(os.listdir(ckpt_folder)) == 2
    else:
        ckpt_folder = os.path.join(run_root, f"{logger.run_id}/artifacts/checkpoints")
        assert len(os.listdir(ckpt_folder)) == 1
        assert os.path.isfile(os.path.join(ckpt_folder, "last.ckpt"))


@mock.patch("pytorch_lightning.loggers.mlflow._MLFLOW_AVAILABLE", return_value=True)
def test_mlflow_log_hyperparams(_, tmpdir):
    """Test that the logger creates the folders and files in the right
    place."""

    mlflow_root = os.path.join(tmpdir, "mlflow/")

    logger = MLFlowLogger("test", save_dir=mlflow_root)
    logger.log_hyperparams(dict(a=1, b=2))

    config = get_config(
        logger._tracking_uri,
        logger.run_id,
        tmpdir,
    )

    assert "a" in config
    assert config["a"] == 1
    assert "b" in config
    assert config["b"] == 2
