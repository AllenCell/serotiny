import inspect
import torch
import torch.nn.functional as F
import torch.optim as opt

import numpy as np
import matplotlib.pyplot as plt

def add_pr_curve_tensorboard(
    logger, classes, class_index, test_probs, test_preds, global_step=0, name="_val"
):
    """
    Takes in a "class_index" and plots the corresponding
    precision-recall curve
    """
    tensorboard_preds = test_preds == class_index
    tensorboard_probs = test_probs[:, class_index]

    logger.experiment.add_pr_curve(
        str(classes[class_index]) + name,
        tensorboard_preds,
        tensorboard_probs,
        global_step=global_step,
    )
    logger.experiment.close()


def index_to_onehot(index, n_classes):
    index = index.long()

    onehot = torch.zeros(len(index), n_classes).type_as(index).float()
    onehot.scatter_(1, index, 1)

    return onehot


def find_optimizer(optimizer_name):
    """
    Given optimizer name, get it from torch.optim
    """
    available_optimizers = []
    for cls_name, cls in opt.__dict__.items():
        if inspect.isclass(cls):
            if issubclass(cls, opt.Optimizer):
                available_optimizers.append(cls_name)

    if optimizer_name in available_optimizers:
        optimizer_class = opt.__dict__[optimizer_name]
    else:
        raise KeyError(
            f"optimizer {optimizer_name} not available, "
            f"options are {available_optimizers}"
        )
    return optimizer_class


def find_lr_scheduler(scheduler_name):
    """
    Given scheduler name, get it from torch.optim.lr_scheduler
    """
    available_schedulers = []
    for cls_name, cls in opt.lr_scheduler.__dict__.items():
        if inspect.isclass(cls):
            if "LR" in cls_name and cls_name[0] != "_":
                available_schedulers.append(cls_name)

    if scheduler_name in available_schedulers:
        scheduler_class = opt.lr_scheduler.__dict__[scheduler_name]
    else:
        raise Exception(
            f"scheduler {scheduler_name} not available, "
            f"options are {available_schedulers}"
        )
    return scheduler_class
