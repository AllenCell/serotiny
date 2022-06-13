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
        latent_loss_backprop_when: dict = None,
        latent_optimizer: torch.optim.Optimizer = torch.optim.Adam,
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
            latent_loss = {x_label: latent_loss_target}
        self.latent_loss = latent_loss_target

        if not isinstance(latent_optimizer, (dict, DictConfig)):
            assert x_label is not None
            latent_optimizer = {x_label: latent_optimizer}
        self.latent_optimizer = latent_loss

        if not isinstance(latent_loss_backprop_when, (dict, DictConfig)):
            assert x_label is not None
            latent_loss_backprop_when = {x_label: latent_loss_backprop_when}
        self.latent_loss_backprop_when = latent_loss

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
                optimizers = self.optimizers()
                main_optim = optimizers.pop(0)
                main_optim.zero_grad()
                self.manual_backward(loss)
                main_optim.step()

            for optim_ix, optim in enumerate(optimizers):
                part = self.latent_optimizer_map[optim_ix]
                loss = self.latent_loss[part](
                    z_parts_params[part], batch[self.latent_loss_target[part]]
                )
                if stage == "train":
                    mod = self.latent_loss_backprop_when.get(part) or 1
                    if batch_idx % mod == 0:
                        optim.zero_grad()
                        self.manual_backward(loss)
                        optim.step()

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
            self.latent_optimizer_map = dict()
            for optim_ix, (key, latent_optim) in enumerate(
                self.latent_optimizer.items()
            ):
                self.latent_optimizer_map[optim_ix] = key
                optimizers.append(
                    latent_optim(
                        chain(
                            self.encoder[key].parameters(),
                            self.latent_loss[key].parameters(),
                        )
                    )
                )

            return optimizers
