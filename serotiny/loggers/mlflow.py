import os
from typing import Union, Dict, Any, Namespace
from pathlib import Path
import tempfile

from omegaconf import OmegaConf
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger as _MLFlowLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only


class MLFlowLogger(_MLFlowLogger):
    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        mode = self.trainer.fn

        with tempfile.TemporaryDirectory() as tmp_dir:
            conf_path = Path(tmp_dir) / f"{mode}.yaml"
            with open(conf_path, "w") as f:
                config = OmegaConf.create(params)
                OmegaConf.save(config=config, f=f)

            self.experiment.log_artifact(
                self.run_id, local_path=conf_path, artifact_path="config"
            )

    def after_save_checkpoint(self, ckpt_callback: ModelCheckpoint) -> None:
        """Called after model checkpoint callback saves a new checkpoint."""

        monitor = ckpt_callback.monitor
        if monitor is not None:
            artifact_path = f"checkpoints/{monitor}"
            existing_ckpts = set(
                _.split("/")[-1]
                for _ in self.experiment.list_artifacts(self.run_id, path=artifact_path)
            )

            top_k_ckpts = set(
                _.split("/")[-1] for _ in ckpt_callback.best_k_models.keys()
            )

            to_delete = existing_ckpts - top_k_ckpts
            to_upload = top_k_ckpts - existing_ckpts

            run = self.experiment.get_run(self.run_id)
            repository = get_artifact_repository(run.info.artifact_uri)
            for ckpt in to_delete:
                repository.delete_artifact(Path(ckpt).name)

            for ckpt in to_upload:
                self.experiment.log_artifact(
                    self.run_id, local_path=ckpt, artifact_path=artifact_path
                )
        else:
            filepath = ckpt_callback.best_model_path
            artifact_path = "checkpoints"

            # mimic ModelCheckpoint's behavior: if `self.save_top_k == 1` only
            # keep the latest checkpoint, otherwise keep all of them.
            if self.save_top_k == 1:
                last_path = Path(filepath).with_name("last.ckpt")
                os.link(filepath, last_path)

                self.experiment.log_artifact(
                    self.run_id, local_path=last_path, artifact_path=artifact_path
                )

                os.unlink(last_path)
            else:
                self.experiment.log_artifact(
                    self.run_id, local_path=filepath, artifact_path=artifact_path
                )
