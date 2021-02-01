#!/usr/bin/env python3

import torch.nn as nn
import pytorch_lightning as pl

def acc_prec_recall(n_classes):
    """
    util function to instantiate a ModuleDict for metrics
    """
    return nn.ModuleDict(
        {
            'accuracy': pl.metrics.Accuracy(),
            'precision': pl.metrics.Precision(
                num_classes=n_classes,
                average='macro'
            ),
            'recall': pl.metrics.Recall(
                num_classes=n_classes,
                average='macro'
            ),
        }
    )
