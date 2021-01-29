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


def tune_model(
    datasets_path: str,
    output_path: str,
    classes: list = ["M0", "M1/M2", "M3", "M4/M5", "M6/M7"],
    model: str = "basic",
    label: str = "Draft mitotic state resolved",
    batch_size: int = 64,
    num_gpus: int = 4,
    num_cpus: int = 30,
    num_workers: int = 50,
    channel_indexes: list = ["dna", "membrane"],
    num_epochs: int = 10,
    lr: int = 0.001,
    model_optimizer: str = "sgd",
    model_scheduler: str = "reduce_lr_plateau",
    id_fields: list = ["CellId", "CellIndex", "FOVId"],
    channels: list = ["membrane", "structure", "dna"],
    num_samples: int = 10,
    gpus_per_trial: int = 1,
    search_space: dict = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128]),
        "model_optimizer": tune.choice(["sgd", "adam"]),
        "model": tune.choice(["basic", "resnet18"])
    },
    **kwargs,
):
    """
    """

    ray.init(num_cpus=num_cpus, num_gpus=num_gpus)

    reporter = CLIReporter(
        parameter_columns=[key for key in search_space.keys()],
        # loss and mean accuracy are specified in the tune_callback
        # passed into the trainer in train_model
        metric_columns=["loss", "mean_accuracy", "training_iteration"]
    )

    config = {
        "datasets_path": datasets_path,
        "output_path": output_path,
        "classes": classes,
        "model": model,
        "label": label,
        "batch_size": batch_size,
        "num_gpus": gpus_per_trial,
        "num_workers": num_workers,
        "channel_indexes": channel_indexes,
        "num_epochs": num_epochs,
        "lr":  lr,
        "model_optimizer": model_optimizer,
        "model_scheduler": model_scheduler,
        "id_fields": id_fields,
        "channels": channels,
        "test": False,
        "tune_bool": True
    }

    config.update(search_space)

    print(config)

    # hyperopt = HyperOptSearch(metric="loss", mode="min")

    tune_scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=1,
        reduction_factor=2)

    analysis = tune.run(
        train_model_config,
        resources_per_trial={
            "cpu": 5,
            "gpu": gpus_per_trial
        },
        config=config,
        # search_alg=hyperopt,
        local_dir=output_path,
        name="ray_logs",
        metric="loss",
        mode="min",
        num_samples=num_samples,
        progress_reporter=reporter,
        scheduler=tune_scheduler
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
    return


if __name__ == '__main__':
    # example command:
    # python -m serotiny.steps.tune_model \
    #     --datasets_path
    #  "/allen/aics/modeling/ritvik/projects/serotiny/results/splits/"
    #     --output_path
    #  "/allen/aics/modeling/ritvik/projects/serotiny/results/model"

    fire.Fire(tune_model)
