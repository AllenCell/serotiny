#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import fire

import ray

# from ray.tune.suggest.hyperopt import HyperOptSearch
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

# from torchvision.utils import save_image

from .train_model import train_model_config

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


def tune_classifier(
    datasets_path: str,
    output_path: str,
    datamodule: str,
    model: str,
    batch_size: int,
    num_gpus: int,
    num_workers: int,
    num_epochs: int,
    lr: float,
    optimizer: str,
    scheduler: str,
    test: bool,
    tune_bool: bool,
    x_label: str,
    y_label: str,
    classes: str,
    num_cpus: int,
    num_samples: int,
    gpus_per_trial: int,
    search_space: dict,
):
    """
    Initialize ray instance with specified number of cpus
    and gpus. Pass in a search space to sample from,
    a number of samples to choose from that space, and
    resources per trial. Other
    training arguments are the same as train_model

    Extra parameters compared to train_classifier
    -----------
    num_cpus: int
        Number of cpus to initialize ray instance with

    num_samples: int
        Number of samples per hyperparam search
        Example: 16

    gpus_per_trials: int
        Example: 1

    search_space: dict
        Parameters to perform hyper param search on
        Example:
        {
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128]),
        "model_optimizer": tune.choice(["sgd", "adam"]),
        }
    """

    # Initialize ray
    ray.init(num_cpus=num_cpus, num_gpus=num_gpus)
    assert ray.is_initialized() is True

    reporter = CLIReporter(
        parameter_columns=[key for key in search_space.keys()],
        # loss and mean accuracy are specified in the tune_callback
        # passed into the trainer in train_model
        metric_columns=["loss", "mean_accuracy", "training_iteration"],
    )

    config = {
        "datasets_path": datasets_path,
        "output_path": output_path,
        "classes": classes,
        "datamodule": datamodule,
        "model": model,
        "batch_size": batch_size,
        "num_gpus": gpus_per_trial,
        "num_workers": num_workers,
        "num_epochs": num_epochs,
        "lr": lr,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "test": False,
        "tune_bool": True,
        "x_label": x_label,
        "y_label": y_label,
    }

    config.update(search_space)

    # hyperopt = HyperOptSearch(metric="loss", mode="min")

    tune_scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

    analysis = tune.run(
        train_classifier_config,
        resources_per_trial={"cpu": 10, "gpu": gpus_per_trial},
        config=config,
        # search_alg=hyperopt,
        local_dir=output_path,
        name="ray_logs",
        metric="loss",
        mode="min",
        num_samples=num_samples,
        progress_reporter=reporter,
        scheduler=tune_scheduler,
    )
    print("Best config: ", analysis.get_best_config)
    print("dataframe", analysis.results_df)
    print("best trial", analysis.best_trial)  # Get best trial
    print("best logdir", analysis.best_logdir)  # Get best trial's logdir
    print("best checkpoint", analysis.best_checkpoint)
    # Get best trial's best checkpoint
    print("best result", analysis.best_result)  # Get best trial's last results
    print("best result df", analysis.best_result_df)
    # Get best result as pandas dataframe

    print("Best hyperparameters found were: ", analysis.best_config)

    # Shutdown Ray
    ray.shutdown()
    assert ray.is_initialized() is False


if __name__ == "__main__":
    # example command:
    # python -m serotiny.steps.tune_model \
    #     --datasets_path
    #  "/allen/aics/modeling/ritvik/projects/serotiny/results/splits/"
    #     --output_path
    #  "/allen/aics/modeling/ritvik/projects/serotiny/results/model"

    fire.Fire(tune_model)
