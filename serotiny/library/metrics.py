# import pytorch_lightning as pl
# import pytorch_lightning.metrics.functional as plm
# import pytorch_lightning.metrics as plm2
# import torch
# import logging

# log = logging.getLogger(__name__)


# class MetricsCalculator(object):
#     def __init__(self, metrics, num_classes=5) -> None:
#         super().__init__()
#         self.metrics = metrics
#         self.num_classes = num_classes

#     def _mean_metrics(self, logs, key):
#         return (
#             logs[key]
#             if isinstance(logs, dict)
#             else torch.stack([item[key] for item in logs]).mean()
#         )

#     def accuracy(self, true, pred):
#         _, class_preds_batch = torch.max(pred, 1)
#         acc = plm2.Accuracy()
#         return acc(class_preds_batch, true)

#     def precision(self, true, pred):
#         _, class_preds_batch = torch.max(pred, 1)
#         prec = plm2.Precision(num_classes=5)
#         return prec(class_preds_batch, true)

#     def average_precision(self, true, pred):
#         _, class_preds_batch = torch.max(pred, 1)
#         return plm.average_precision(class_preds_batch, true)

#     def recall(self, true, pred):
#         _, class_preds_batch = torch.max(pred, 1)
#         prec = plm2.Recall(num_classes=5)
#         return prec(class_preds_batch, true)

#     def fbeta(self, true, pred):
#         _, class_preds_batch = torch.max(pred, 1)
#         fbeta = pl.metrics.classification.Fbeta(beta=0.5)
#         return fbeta(class_preds_batch, true)

#     def confusion_matrix(self, true, pred):
#         _, class_preds_batch = torch.max(pred, 1)
#         return plm.confusion_matrix(
#             class_preds_batch, true, num_classes=self.num_classes
#         )

#     def dice_score(self, true, pred):
#         _, class_preds_batch = torch.max(pred, 1)
#         return plm.dice_score(pred, true)

#     def f1_score(self, true, pred):
#         _, class_preds_batch = torch.max(pred, 1)
#         return plm.f1_score(class_preds_batch, true)

#     def generate_logs(self, loss, preds, true, prefix, with_loss=True):
#         if with_loss:
#             return {
#                 f"{prefix}_loss": loss,
#                 **{
#                     f"{prefix}_{key}": MetricsCalculator.__dict__[key](
#                         self, true, preds
#                     )
#                     for key in self.metrics
#                 },
#             }
#         else:
#             return {
#                 **{
#                     f"{prefix}_{key}": MetricsCalculator.__dict__[key](
#                         self, true, preds
#                     )
#                     for key in self.metrics
#                 },
#             }

#     def generate_mean_metrics(self, outputs, prefix):
#         mean_keys = ["loss"] + self.metrics
#         return {
#             f"{prefix}_{key}": self._mean_metrics(outputs, f"{prefix}_{key}")
#             for key in mean_keys
#         }
