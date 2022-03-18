import inspect

import torch.optim as opt


def find_optimizer(optimizer_name):
    """Given optimizer name, get it from torch.optim."""
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
    """Given scheduler name, get it from torch.optim.lr_scheduler."""
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
