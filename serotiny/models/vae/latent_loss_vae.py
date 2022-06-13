from typing import Optional, Sequence
from itertools import chain
from omegaconf import DictConfig

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss as Loss

from .base_vae import BaseVAE
from .priors import Prior


class LatentLossVAE(BaseVAE):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        latent_dim: int,
        x_label: str,
        latent_loss: dict,
        latent_loss_target: dict,
        latent_loss_weights: dict,
        latent_loss_backprop_when: dict = None,
        latent_loss_optimizer: torch.optim.Optimizer = torch.optim.Adam,
        latent_loss_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        beta: float = 1.0,
        id_label: Optional[str] = None,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        loss_mask_label: Optional[str] = None,
        reconstruction_loss: Loss = nn.MSELoss(reduction="none"),
        prior: Optional[Sequence[Prior]] = None,
        cache_outputs: Sequence = ("test",),
        **kwargs,
    ):
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            latent_dim=latent_dim,
            x_label=x_label,
            beta=beta,
            id_label=id_label,
            optimizer=optimizer,
            loss_mask_label=loss_mask_label,
            reconstruction_loss=reconstruction_loss,
            prior=prior,
            cache_outputs=cache_outputs,
            **kwargs,
        )

        if not isinstance(latent_loss, (dict, DictConfig)):
            assert x_label is not None
            latent_loss = {x_label: latent_loss}
        self.latent_loss = latent_loss

        if not isinstance(latent_loss_target, (dict, DictConfig)):
            assert x_label is not None
            latent_loss_target = {x_label: latent_loss_target}
        self.latent_loss_target = latent_loss_target

        if not isinstance(latent_loss_weights, (dict, DictConfig)):
            assert x_label is not None
            latent_loss_weights = {x_label: latent_loss_weights}
        self.latent_loss_weights = latent_loss_weights

        if not isinstance(latent_loss_backprop_when, (dict, DictConfig)):
            assert x_label is not None
            latent_loss_backprop_when = {x_label: latent_loss_backprop_when}
        self.latent_loss_backprop_when = latent_loss

        if not isinstance(latent_loss_optimizer, (dict, DictConfig)):
            assert x_label is not None
            latent_loss_optimizer = {x_label: latent_loss_optimizer}
        self.latent_loss_optimizer = latent_loss_optimizer

        if not isinstance(latent_loss_optimizer, (dict, DictConfig)):
            assert x_label is not None
            latent_loss_scheduler = {x_label: latent_loss_scheduler}
        self.latent_loss_scheduler = latent_loss_scheduler

        self.automatic_optimization = False

        def _step(self, stage, batch, batch_idx, logger):
            (
                x_hat,
                z_parts,
                z_parts_params,
                loss,
                reconstruction_loss,
                kld_loss,
                kld_per_part,
            ) = self.forward(batch, decode=True, compute_loss=True)

            if stage == "train":
                optimizers, lr_schedulers = self.optimizers()
                main_optim = optimizers.pop(0)
                main_lr_sched = lr_schedulers.pop(0)
                main_optim.zero_grad()
                self.manual_backward(loss)
                main_optim.step()
                if main_lr_sched is not None:
                    main_lr_sched.step()

            for optim_ix, (optim, lr_sched) in enumerate(
                zip(optimizers, lr_schedulers)
            ):
                part = self.latent_loss_optimizer_map[optim_ix]
                loss = self.latent_loss[part](
                    z_parts_params[part], batch[self.latent_loss_target[part]]
                )
                loss = loss * self.latent_loss_weights.get(part, 1.0)

                if stage == "train":
                    mod = self.latent_loss_backprop_when.get(part) or 1
                    if batch_idx % mod == 0:
                        optim.zero_grad()
                        self.manual_backward(loss)
                        optim.step()
                        if lr_sched is not None:
                            lr_sched.step()

                on_step = stage == "train"
                self.log(
                    f"{stage}_{part}_latent_loss",
                    loss.detach().cpu(),
                    logger=logger,
                    on_step=on_step,
                    on_epoch=True,
                )

            self.log_metrics(stage, reconstruction_loss, kld_loss, loss, logger)

            return self.make_results_dict(
                stage,
                batch,
                loss,
                reconstruction_loss,
                kld_loss,
                kld_per_part,
                z_parts,
                z_parts_params,
                x_hat,
            )

        def configure_optimizers(self):
            optimizers = [self.optimizer(self.parameters())]

            lr_schedulers = [
                (
                    self.lr_scheduler(optimizer=optimizers[0])
                    if self.lr_scheduler is not None
                    else None
                )
            ]

            self.latent_loss_optimizer_map = dict()
            for optim_ix, (part, latent_optim) in enumerate(
                self.latent_loss_optimizer.items()
            ):
                self.latent_loss_optimizer_map[optim_ix] = part
                optimizers.append(
                    latent_optim(
                        chain(
                            self.encoder[part].parameters(),
                            self.latent_loss[part].parameters(),
                        )
                    )
                )
                lr_sched = self.latent_loss_scheduler.get(part)
                if lr_sched is not None:
                    lr_sched = lr_sched(optimizer=optimizers[-1])
                lr_schedulers.append(lr_sched)

            return optimizers, lr_schedulers
