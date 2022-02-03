import os
import datetime
import tempfile
from pathlib import Path

from omegaconf import OmegaConf
import pytorch_lightning as pl
from mlflow.utils.autologging_utils import autologging_integration, safe_patch

import mlflow
from mlflow.tracking import MlflowClient

@autologging_integration("pytorch")
def patched_autolog(
    log_every_n_epoch=1,
    log_models=True,
    disable=False,
    exclusive=False,
    disable_for_unsupported_versions=False,
    silent=False,
    fit_or_test="fit",
):
    """
    Patched mlflow autolog, to enable on_test_epoch_end callbacks and to
    cover trainer.test calls as well
    """
    from pytorch_lightning.utilities import rank_zero_only
    import mlflow.pytorch._pytorch_autolog as _pytorch_autolog
    from mlflow.utils.autologging_utils import safe_patch

    @rank_zero_only
    def _patched_on_test_epoch_end(
            original, self, trainer, pl_module, *args
    ):  # pylint: disable=signature-differs,arguments-differ,unused-argument
        self._log_metrics(trainer, pl_module)

    assert fit_or_test in ["fit", "test"]

    safe_patch("pytorch", _pytorch_autolog.__MLflowPLCallback,
               "on_test_epoch_end", _patched_on_test_epoch_end)

    safe_patch("pytorch", mlflow.pytorch, "autolog", patched_autolog)

    safe_patch("pytorch", pl.Trainer, fit_or_test, _pytorch_autolog.patched_fit,
               manage_run=True)

    safe_patch("pytorch", pl.Trainer, "save_checkpoint",
               _patched_save_checkpoint)


def _is_empty(conf, key):
    return conf.get(key, None) is None


def _validate_mlflow_conf(conf):
    if _is_empty(conf, "tracking_uri"):
        raise ValueError("Must specify `tracking_uri`")

    if _is_empty(conf, "experiment_name") and _is_empty(conf, "run_id"):
        raise ValueError("If you don't specify `experiment_name`, you must "
                         "specify `run_id`.")

    if _is_empty(conf, "run_name") and _is_empty(conf, "run_id"):
        raise ValueError("You must specify at least `run_id` or `run_name`")


def _patched_save_checkpoint(original, self, filepath, save_weights_only):
    original(self, filepath, save_weights_only)

    latest_path = Path(filepath).with_name("latest.ckpt")
    os.link(filepath, latest_path)

    mlflow.log_artifact(
        local_path=latest_path,
        artifact_path="checkpoints"
    )
    os.unlink(latest_path)


def _get_latest_checkpoint(tracking_uri, run_id, tmp_dir):
    client = MlflowClient(tracking_uri=tracking_uri)

    return client.download_artifacts(
        run_id=run_id,
        path="checkpoints/latest.ckpt",
        dst_path=tmp_dir
    )


def _mlflow_prep(mlflow_conf, trainer, model, data, fit_or_test):
    _validate_mlflow_conf(mlflow_conf)
    assert fit_or_test in ["fit", "test"]

    # if autolog arguments aren't given, or are None, set to empty dict
    autolog = (OmegaConf.to_object(mlflow_conf.autolog)
               if hasattr(mlflow_conf, "autolog") else None)
    autolog = (autolog if autolog is not None else {})
    autolog["fit_or_test"] = fit_or_test
    if fit_or_test == "test":
        autolog["log_models"] = False
    patched_autolog(**autolog)

    mlflow.set_tracking_uri(mlflow_conf.tracking_uri)

    run_id = mlflow_conf.get("run_id", None)
    if run_id is None:
        run_name = mlflow_conf.get("run_name", None)
        assert run_name is not None
        # creates experiment if it doesn't exist, otherwise just gets it
        experiment = mlflow.set_experiment(
            experiment_name=mlflow_conf.experiment_name)

        runs = [
            run for run in mlflow.list_run_infos(experiment_id=experiment.id)
            if run.name == run_name
        ]

        if len(runs) > 1:
            raise ValueError("You provided `run_name`, but there are multiple "
                             "runs in this experiment with that name. Please "
                             "specify `run_id`.")

        run_id = runs[0].id

    else:
        run = mlflow.get_run(run_id=run_id)
        experiment = mlflow.set_experiment(
            experiment_id=run.info.experiment_id)

    return experiment, run_id


def mlflow_fit(mlflow_conf, trainer, model, data):
    experiment, run_id = _mlflow_prep(
        mlflow_conf, trainer, model, data, "fit")

    # if run_id has been specified, we're trying to resume
    if run_id is not None and not trainer.checkpoint_callback:
        raise ValueError("You're trying to resume training, but "
                         "checkpointing is not enabled.")

    with mlflow.start_run(
            experiment_id=experiment.experiment_id,
            run_name=mlflow_conf.run_name,
            run_id=run_id,
            nested=(run_id is not None)):

        # we don't know yet if there's a checkpoint
        ckpt_path = None

        with tempfile.TemporaryDirectory() as tmp_dir:
            if run_id is not None:
                ckpt_path = _get_latest_checkpoint(mlflow_conf.tracking_uri,
                                                   run_id,
                                                   tmp_dir)

            trainer.fit(model, data, ckpt_path=ckpt_path)


def mlflow_apply(mlflow_conf, trainer, model, data):
    experiment, run_id = _mlflow_prep(
        mlflow_conf, trainer, model, data, "test")

    if run_id is None:
        raise ValueError("You're calling serotiny apply but you "
                         "haven't specified the run_id")

    with mlflow.start_run(
            experiment_id=experiment.experiment_id,
            run_name=mlflow_conf.run_name,
            run_id=run_id,
            nested=(run_id is not None)):

        # we don't know yet if there's a checkpoint
        ckpt_path = None

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = _get_latest_checkpoint(mlflow_conf.tracking_uri,
                                               run_id,
                                               tmp_dir)

            trainer.test(model, data, ckpt_path=ckpt_path)
        mlflow.end_run(status="FINISHED")
