import os
import datetime
import tempfile
from pathlib import Path

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

    latest_path = Path(filepath).with_name("latest.ckpt")
    os.link(filepath, latest_path)

    mlflow.log_artifact(
        local_path=latest_path,
        artifact_path="checkpoints"
    )
    os.unlink(latest_path)


def _timestamp():
    return datetime.datetime.now().strftime("%d_%m_%Y__%H_%M_%S")


def _parse_timestamp(ts):
    return datetime.datetime.strptime(ts, "%d_%m_%Y__%H_%M_%S")


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


def _mlflow_prep(mlflow_conf, trainer, model, data):
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

    run_id = mlflow_conf.get("run_id", None)

    return experiment, run_id, ckpt_path


def mlflow_fit(mlflow_conf, trainer, model, data):
    experiment, run_id, ckpt_path = _mlflow_prep(
        mlflow_conf, trainer, model, data)

    # if run_id has been specified, we're trying to resume
    if run_id is not None and not trainer.checkpoint_callback:
        raise ValueError("You're trying to resume training, but "
                         "checkpointing is not enabled.")

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

            trainer.fit(model, data, ckpt_path=ckpt_path)


def mlflow_apply(mlflow_conf, trainer, model, data):
    experiment, run_id, ckpt_path = _mlflow_prep(
        mlflow_conf, trainer, model, data)

    if run_id is None:
        raise ValueError("You're calling serotiny apply but you "
                         "haven't specified the run_id")

    with mlflow.start_run(
            experiment_id=experiment.experiment_id,
            run_name=mlflow_conf.run_name,
            run_id=run_id,
            nested=(run_id is not None)):

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = _get_newest_checkpoint(mlflow_conf.tracking_uri,
                                               run_id,
                                               tmp_dir)

            trainer.test(model, data, ckpt_path=ckpt_path)
