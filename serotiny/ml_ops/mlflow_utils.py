import logging
import os
import tempfile
from pathlib import Path
from contextlib import contextmanager

import mlflow
from hydra._internal.utils import _locate
from mlflow.tracking import MlflowClient
from mlflow.utils.autologging_utils import autologging_integration
from omegaconf import OmegaConf

from .utils import flatten_config, save_model_predictions
from .ml_ops import instantiate

logger = logging.getLogger(__name__)


def _log_metrics_step(self, trainer, model):
    sanity_checking = trainer.sanity_checking
    if sanity_checking:
        return

    if (trainer.global_step + 1) % trainer.log_every_n_steps == 0:
        # `trainer.callback_metrics` contains both training and validation metrics
        cur_metrics = trainer.callback_metrics
        # Cast metric value as  float before passing into logger.
        metrics = dict(
            map(lambda x: ("batch/" + x[0], float(x[1])), cur_metrics.items())
        )

        self.metrics_logger.record_metrics(metrics, trainer.global_step)


@autologging_integration("pytorch")
def patched_autolog(
    log_every_n_epoch=1,
    log_models=False,  # we handle this
    disable=False,
    exclusive=False,
    disable_for_unsupported_versions=False,
    silent=False,
    mode="fit",
):
    """Patched mlflow autolog, to enable on_test_epoch_end callbacks and to
    cover trainer.test calls as well."""

    import mlflow.pytorch._pytorch_autolog as _pytorch_autolog
    from mlflow.utils.autologging_utils import safe_patch
    from pytorch_lightning import Trainer
    from pytorch_lightning.utilities import rank_zero_only

    @rank_zero_only
    def _patched_on_test_epoch_end(
        original, self, trainer, model, *args
    ):  # pylint: disable=signature-differs,arguments-differ,unused-argument
        self._log_metrics(trainer, model)

    @rank_zero_only
    def _patched_on_train_batch_end(original, self, trainer, model, *args):
        _log_metrics_step(self, trainer, model)

    assert mode in ["fit", "test", "predict"]

    safe_patch(
        "pytorch",
        _pytorch_autolog.__MLflowPLCallback,
        "on_test_epoch_end",
        _patched_on_test_epoch_end,
    )

    safe_patch(
        "pytorch",
        _pytorch_autolog.__MLflowPLCallback,
        "on_train_batch_end",
        _patched_on_train_batch_end,
    )

    safe_patch("pytorch", mlflow.pytorch, "autolog", patched_autolog)

    safe_patch("pytorch", Trainer, mode, _pytorch_autolog.patched_fit, manage_run=True)

    safe_patch("pytorch", Trainer, "save_checkpoint", _patched_save_checkpoint)


def _is_empty(conf, key):
    return conf.get(key, None) is None


def _validate_mlflow_conf(conf):
    if _is_empty(conf, "tracking_uri"):
        raise ValueError("Must specify `tracking_uri`")

    if _is_empty(conf, "experiment_name") and _is_empty(conf, "run_id"):
        raise ValueError(
            "If you don't specify `experiment_name`, you must " "specify `run_id`."
        )

    if _is_empty(conf, "run_name") and _is_empty(conf, "run_id"):
        raise ValueError("You must specify at least `run_id` or `run_name`")


def _patched_save_checkpoint(original, self, filepath, save_weights_only):
    original(self, filepath, save_weights_only)

    latest_path = Path(filepath).with_name("latest.ckpt")
    os.link(filepath, latest_path)

    mlflow.log_artifact(local_path=latest_path, artifact_path="checkpoints")
    os.unlink(latest_path)


def _get_latest_checkpoint(tracking_uri, run_id, tmp_dir):
    client = MlflowClient(tracking_uri=tracking_uri)

    try:
        return client.download_artifacts(
            run_id=run_id, path="checkpoints/latest.ckpt", dst_path=tmp_dir
        )
    except:  # noqa
        return None


def _get_config(tracking_uri, run_id, tmp_dir, mode="test"):
    client = MlflowClient(tracking_uri=tracking_uri)

    try:
        return client.download_artifacts(
            run_id=run_id, path=f"configs/{mode}_config.yaml", dst_path=tmp_dir
        )
    except:  # noqa
        return None


def load_model_from_checkpoint(tracking_uri, run_id):
    mlflow.set_tracking_uri(tracking_uri)
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = _get_latest_checkpoint(tracking_uri, run_id, tmp_dir)
        if ckpt_path is None:
            logger.info("Given run_id doesn't have a checkpoint we can use.")
            return

        config = _get_config(tracking_uri, run_id, tmp_dir, mode="train")
        config = OmegaConf.load(config)
        config = OmegaConf.to_container(config, resolve=True)

        model_conf = config["model"]
        model_class = model_conf.pop("_target_")
        model_conf = instantiate(model_conf)
        model_class = _locate(model_class)
        return model_class.load_from_checkpoint(ckpt_path, **model_conf)


def _get_patience(trainer):
    from pytorch_lightning.callbacks import EarlyStopping

    for callback in trainer.callbacks:
        if isinstance(callback, EarlyStopping):
            return callback.patience
    return None


def _mlflow_prep(mlflow_conf, trainer, model, data, mode):
    logger.info("Validating and processing MLFlow configuration")

    _validate_mlflow_conf(mlflow_conf)
    assert mode in ["fit", "test", "predict"]

    # if autolog arguments aren't given, or are None, set to empty dict
    autolog = (
        OmegaConf.to_object(mlflow_conf.autolog)
        if hasattr(mlflow_conf, "autolog")
        else None
    )
    autolog = autolog if autolog is not None else {}
    autolog["mode"] = mode
    if mode == "test":
        autolog["log_models"] = False
    patched_autolog(**autolog)

    mlflow.set_tracking_uri(mlflow_conf.tracking_uri)

    run_id = mlflow_conf.get("run_id", None)
    if run_id is None:
        run_name = mlflow_conf.get("run_name", None)
        assert run_name is not None
        # creates experiment if it doesn't exist, otherwise just gets it
        experiment = mlflow.set_experiment(experiment_name=mlflow_conf.experiment_name)

        runs = []
        for run_info in mlflow.list_run_infos(experiment_id=experiment.experiment_id):
            run_tags = mlflow.get_run(run_info.run_id).data.tags

            if "mlflow.runName" in run_tags:
                if run_tags["mlflow.runName"] == run_name:
                    runs.append(run_info.run_id)

        if len(runs) > 1:
            raise ValueError(
                "You provided `run_name`, but there are multiple "
                "runs in this experiment with that name. Please "
                "specify `run_id`."
            )
        elif len(runs) == 1:
            run_id = runs[0]
        else:  # redundant but leaving it here to be explicit
            run_id = None

    else:
        run = mlflow.get_run(run_id=run_id)
        experiment = mlflow.set_experiment(experiment_id=run.info.experiment_id)

    patience = _get_patience(trainer)

    # before this point, argv[0] will contain e.g. "/path/to/exectuable/serotiny train"
    # because we append "train" or "test" for presentation purposes in the CLI.
    # however, this causes problems when using tools that need to fork this program
    # e.g. torch's ddp training strategy, and which make use of argv[0] to rerun
    # the program. For that reason, from this point on we remove the suffix, and
    # re-insert it as the second element in argv

    try:
        import sys

        command, suffix = sys.argv[0].split(" ")
        sys.argv[0] = command
        sys.argv.insert(1, suffix)
    except:  # noqa
        # in hydra sweeps, this only needs to be done once and will cause an
        # error if we try to do it again
        pass

    return experiment, run_id, patience


def _log_conf(tmp_dir, full_conf, mode):
    if mode not in ("train", "test", "predict"):
        raise ValueError(f"Got unexpected mode: '{mode}")

    conf_path = Path(tmp_dir) / f"{mode}_config.yaml"
    with open(conf_path, "w") as f:
        OmegaConf.save(config=full_conf, f=f)
        mlflow.log_artifact(local_path=conf_path, artifact_path="configs")


def _log_predictions(preds_dir, tracking_uri, run_id):
    client = MlflowClient(tracking_uri=tracking_uri)
    client.log_artifacts(
        run_id=run_id, local_dir=preds_dir, artifact_path="predictions"
    )


def mlflow_fit(mlflow_conf, trainer, model, data, full_conf, test=False):
    experiment, run_id, patience = _mlflow_prep(
        mlflow_conf, trainer, model, data, "fit"
    )

    # if run_id has been specified, we're trying to resume
    if run_id is not None and not trainer.checkpoint_callback:
        raise ValueError(
            "You're trying to resume training, but checkpointing is not enabled."
        )

    skip = False
    with mlflow.start_run(
        experiment_id=experiment.experiment_id,
        run_name=mlflow_conf.run_name,
        run_id=run_id,
        nested=(run_id is not None),
    ):

        if run_id is None:
            params_to_log = mlflow_conf.get("params_to_log", None)
            if params_to_log is not None:
                mlflow.log_params(
                    flatten_config(OmegaConf.to_container(params_to_log, resolve=True))
                )

        if run_id is not None:
            client = MlflowClient(tracking_uri=mlflow_conf["tracking_uri"])
            run = client.get_run(run_id)
            if patience is not None:
                if "wait_count" in run.data.metrics:
                    if run.data.metrics["wait_count"] >= patience:
                        logger.info(
                            "This model has trained until or beyond the early "
                            "stopping patience value. Skipping"
                        )
                        skip = True

            if trainer.max_epochs is not None:
                if "train_loss" in run.data.metrics:
                    timepoints = client.get_metric_history(run_id, "train_loss")
                    if len(timepoints) >= trainer.max_epochs:
                        logger.info(
                            "This model has trained until or beyond "
                            "max epochs. Skipping"
                        )
                        skip = True

        if not skip:
            # we don't know yet if there's a checkpoint
            ckpt_path = None
            with tempfile.TemporaryDirectory() as tmp_dir:
                if run_id is not None:
                    logger.info("Trying to retrieve checkpoint to resume training")
                    ckpt_path = _get_latest_checkpoint(
                        mlflow_conf.tracking_uri, run_id, tmp_dir
                    )

                _log_conf(tmp_dir, full_conf, "train")

                logger.info("Calling trainer.fit")
                trainer.fit(model, data, ckpt_path=ckpt_path)

                if test:
                    logger.info("Calling trainer.test")
                    trainer.test(model, data)

        mlflow.end_run(status="FINISHED")


def mlflow_test(mlflow_conf, trainer, model, data, full_conf):
    experiment, run_id, patience = _mlflow_prep(
        mlflow_conf, trainer, model, data, "test"
    )

    if run_id is None:
        raise ValueError(
            "You're calling serotiny test but you " "haven't specified the run_id"
        )

    with mlflow.start_run(
        experiment_id=experiment.experiment_id,
        run_name=mlflow_conf.run_name,
        run_id=run_id,
        nested=(run_id is not None),
    ):

        # we don't know yet if there's a checkpoint
        ckpt_path = None

        with tempfile.TemporaryDirectory() as tmp_dir:
            if _get_config(
                mlflow_conf.tracking_uri, run_id, tmp_dir
            ) is not None and not full_conf.get("force", False):
                logger.info(
                    "This run has been tested before. If you want to "
                    "retest it, set ++force=True"
                )
            else:
                ckpt_path = _get_latest_checkpoint(
                    mlflow_conf.tracking_uri, run_id, tmp_dir
                )

                _log_conf(tmp_dir, full_conf, "test")

                if ckpt_path is None:
                    logger.info("No checkpoint found for this run. Skipping.")
                else:
                    logger.info("Calling trainer.test")
                    trainer.test(model, data, ckpt_path=ckpt_path)
        mlflow.end_run(status="FINISHED")


def mlflow_predict(mlflow_conf, trainer, model, data, full_conf):
    experiment, run_id, patience = _mlflow_prep(
        mlflow_conf, trainer, model, data, "predict"
    )

    if run_id is None:
        raise ValueError(
            "You're calling serotiny predict but you " "haven't specified the run_id"
        )

    with mlflow.start_run(
        experiment_id=experiment.experiment_id,
        run_name=mlflow_conf.run_name,
        run_id=run_id,
        nested=(run_id is not None),
    ):

        # we don't know yet if there's a checkpoint
        ckpt_path = None

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = _get_latest_checkpoint(
                mlflow_conf.tracking_uri, run_id, tmp_dir
            )

            _log_conf(tmp_dir, full_conf, "predict")

            if ckpt_path is None:
                logger.info("No checkpoint found for this run. Skipping.")
            else:
                logger.info("Calling trainer.predict")
                preds = trainer.predict(model, data, ckpt_path=ckpt_path)

                preds_dir = Path(tmp_dir) / "predictions"
                preds_dir.mkdir(exist_ok=True)

                save_model_predictions(model, preds, preds_dir)
                _log_predictions(preds_dir, mlflow_conf["tracking_uri"], run_id)
        mlflow.end_run(status="FINISHED")


@contextmanager
def upload_artifacts(artifact_path, exclude=[], globstr="*"):
    """Util context manager to upload artifacts.

    It yields a temporary directory to store results and then uploads the files
    stored therein. Optionally exclude given files or restrict to a given glob
    filter
    """

    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            yield tmp_dir
        finally:
            for fpath in Path(tmp_dir).glob(globstr):
                if fpath in exclude or str(fpath) in exclude or not fpath.is_file():
                    continue

                mlflow.log_artifact(local_path=fpath, artifact_path=artifact_path)


@contextmanager
def download_artifact(artifact_path):
    """Util context manager to download artifacts.

    Returns a path to the downloaded artifact
    """

    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            client = mlflow.tracking.MlflowClient(mlflow.get_tracking_uri())
            yield client.download_artifacts(
                run_id=mlflow.active_run().info.run_id,
                path=artifact_path,
                dst_path=tmp_dir,
            )

        finally:
            pass
