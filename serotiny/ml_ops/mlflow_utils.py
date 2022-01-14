import time
import tempfile

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.utils.autologging_utils import safe_patch

import pytorch_lightning as pl


def _validate_mlflow_conf(mlflow_conf):
    if (
        (not hasattr(mlflow_conf, "tracking_uri")) or
        (not hasattr(mlflow_conf, "experiment_name")) or
        (mlflow_conf.tracking_uri is None) or
        (mlflow_conf.experiment_name is None)
    ):
        raise ValueError(
            "Must specify `tracking_uri` and `experiment_name`")

    if (
        (not hasattr(mlflow_conf, "run_name") or mlflow_conf.run_name is None) and
        (not hasattr(mlflow_conf, "run_id") or mlflow_conf.run_id is None)
    ):
        raise ValueError(
            "Must specify at least `run_id` or `run_name`.")


def _patched_save_checkpoint(original, self, filepath, save_weights_only):
    original(self, filepath, save_weights_only)

    mlflow.log_artifact(
        local_path=filepath,
        artifact_path=f"checkpoints/{time.time()}"
    )


def _get_newest_checkpoint(tracking_uri, run_id, tmp_dir):
    paths = []
    client = MlflowClient(tracking_uri=tracking_uri)

    for artifact in client.list_artifacts(run_id, path="checkpoints"):
        paths.append(artifact.path.split("/")[-1])

    if len(paths) == 0:
        return None

    path = sorted(paths, key=float, reverse=True)[0]

    ckpt = client.list_artifacts(run_id, path=f"checkpoints/{path}")
    assert len(ckpt) <= 1

    if len(ckpt) == 0:
        return None

    ckpt = ckpt[0].path

    return client.download_artifacts(
        run_id=run_id,
        path=ckpt,
        dst_path=tmp_dir
    )


def mlflow_fit(mlflow_conf, trainer, model, data):
    _validate_mlflow_conf(mlflow_conf)

    mlflow.set_tracking_uri(mlflow_conf.tracking_uri)

    # if autolog arguments aren't given, or are None, set to empty dict
    autolog = (mlflow_conf.autolog if hasattr(mlflow_conf, "w") else None)
    autolog = (autolog if autolog is not None else {})
    mlflow.pytorch.autolog(**autolog)
    safe_patch("pytorch", pl.Trainer, "save_checkpoint",
               _patched_save_checkpoint)

    # creates experiment if it doesn't exist, otherwise just gets it
    experiment = mlflow.set_experiment(
        experiment_name=mlflow_conf.experiment_name)

    # we don't know yet if there's a checkpoint
    ckpt_path = None

    # if run_id has been specified, we're trying to resume
    if hasattr(mlflow_conf, "run_id") and mlflow_conf.run_id is not None:
        run_id = mlflow_conf.run_id
        if not trainer.checkpoint_callback:
            raise ValueError("You're trying to resume training, but "
                             "checkpointing is not enabled.")
    else:
        run_id = None

    with mlflow.start_run(
            experiment_id=experiment.experiment_id,
            run_name=mlflow_conf.run_name,
            run_id=run_id,
            nested=(run_id is not None)):

        with tempfile.TemporaryDirectory() as tmp_dir:
            if run_id is not None:
                ckpt_path = _get_newest_checkpoint(mlflow_conf.tracking_uri,
                                                   run_id,
                                                   tmp_dir)

            if ckpt_path is not None:
                trainer.fit(model, data, ckpt_path=ckpt_path)
