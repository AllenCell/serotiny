from typing import Optional, Sequence
from itertools import chain
from omegaconf import DictConfig

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss as Loss
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from .base_vae import BaseVAE
from .priors import Prior
import torch.nn.functional as F


class LatentLossVAE(BaseVAE):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        latent_dim: int,
        x_label: str,
        x_dim: str,
        output_dim: str,
        continuous_labels: list,
        discrete_labels: list,
        latent_loss: dict,
        latent_loss_target: dict,
        latent_loss_weights: dict,
        latent_loss_backprop_when: dict = None,
        prior: dict = None,
        latent_loss_optimizer: torch.optim.Optimizer = torch.optim.Adam,
        latent_loss_scheduler: LRScheduler = torch.optim.lr_scheduler.StepLR,
        beta: float = 1.0,
        id_label: Optional[str] = None,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler = torch.optim.lr_scheduler.StepLR,
        loss_mask_label: Optional[str] = None,
        reconstruction_loss: torch.nn.modules.loss._Loss = nn.MSELoss(reduction="none"),
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
            lr_scheduler=lr_scheduler,
            loss_mask_label=loss_mask_label,
            reconstruction_loss=reconstruction_loss,
            prior=prior,
            cache_outputs=cache_outputs,
            **kwargs,
        )

        self.continuous_labels = continuous_labels
        self.discrete_labels = discrete_labels
        self.comb_label = self.continuous_labels[-1] + f"_{self.continuous_labels[0]}"

        if not isinstance(latent_loss, (dict, DictConfig)):
            assert x_label is not None
            latent_loss = {x_label: latent_loss}
        self.latent_loss = latent_loss
        self.latent_loss = nn.ModuleDict(latent_loss)

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
        self.latent_loss_backprop_when = latent_loss_backprop_when

        if not isinstance(latent_loss_optimizer, (dict, DictConfig)):
            assert x_label is not None
            latent_loss_optimizer = {x_label: latent_loss_optimizer}
        self.latent_loss_optimizer = latent_loss_optimizer

        if not isinstance(latent_loss_optimizer, (dict, DictConfig)):
            assert x_label is not None
            latent_loss_scheduler = {x_label: latent_loss_scheduler}
        self.latent_loss_scheduler = latent_loss_scheduler

        self.automatic_optimization = False

    def parse_batch(self, batch):

        batch[self.hparams.x_label] = batch[self.hparams.x_label].float() 
        batch[self.comb_label] = batch[self.comb_label].float()
        discrete_batches = []
        for i in self.discrete_labels:
            if len(batch[i].shape) != 2:
                batch[i] = batch[i].view(-1,1)
        return batch

    def encode(self, batch):
        encoded = dict()
        encoded[self.hparams.x_label] = self.encoder[self.hparams.x_label](batch[self.hparams.x_label])
        for part in self.discrete_labels:
            encoded[part] = self.encoder[part](batch[part].argmax(1))
        
        encoded[self.continuous_labels[-1]] = self.encoder[self.continuous_labels[0]](batch[self.comb_label]) @ self.encoder[self.continuous_labels[-1]].weight
        return encoded

    def latent_compose_function(self, z_parts, **kwargs):
        latent_basal = z_parts[self.hparams.x_label]
        latent_perturbation = z_parts[self.continuous_labels[-1]]
        latent_covariate = 0
        for i in self.discrete_labels:
            latent_covariate += z_parts[i]
        return latent_basal + latent_perturbation + latent_covariate

    # def _step(self, stage, batch, batch_idx, logger):
    #     (
    #         x_hat,
    #         z_parts,
    #         z_parts_params,
    #         z_composed,
    #         loss,
    #         reconstruction_loss,
    #         kld_loss,
    #         kld_per_part,
    #     ) = self.forward(batch, decode=True, compute_loss=True)
    #     if stage == 'train':
    #         print(reconstruction_loss, z_composed.max(), batch['drug_dose'].unique())
    #         # import ipdb
    #         # ipdb.set_trace()
    #     _loss = {}
    #     for part in self.latent_loss.keys():
    #         if isinstance(self.latent_loss[part].loss, torch.nn.modules.loss.BCEWithLogitsLoss):
    #             batch[self.latent_loss_target[part]] = batch[self.latent_loss_target[part]].gt(0).float()

    #         _loss[part] = self.latent_loss[part](
    #             z_parts_params[self.hparams.x_label], batch[self.latent_loss_target[part]]
    #         ) * self.latent_loss_weights.get(part, 1.0) 

    #     # import ipdb
    #     # ipdb.set_trace()
    #     optimizers = self.optimizers()
    #     lr_schedulers = self.lr_schedulers()
    #     main_optim = optimizers.pop(0)
    #     main_lr_sched = lr_schedulers.pop(0)

    #     adversarial_flag = False
    #     for optim_ix, (optim, lr_sched) in enumerate(
    #         zip(optimizers[:1], lr_schedulers[:1])
    #     ):
    #         part = self.latent_loss_optimizer_map[optim_ix]

    #         if stage == "train":
    #             mod = self.latent_loss_backprop_when.get(part) or 5

    #             if batch_idx % mod == 0:
    #                 adversarial_flag = True
    #                 optim.zero_grad()
    #                 self.manual_backward(_loss[part])
    #                 optim.step()
    #                 if lr_sched is not None:
    #                     lr_sched.step()

    #         on_step = stage == "train"
    #         self.log(
    #             f"{stage}_{part}_latent_loss",
    #             _loss[part].detach().cpu(),
    #             logger=logger,
    #             on_step=on_step,
    #             on_epoch=True,
    #         )

    #     if (stage == "train") & (adversarial_flag == False):
    #         main_optim.zero_grad()
    #         self.manual_backward(loss - sum(_loss.values()))
    #         main_optim.step()
    #         if main_lr_sched is not None:
    #             main_lr_sched.step()

    #     self.log_metrics(stage, reconstruction_loss, kld_loss, loss, logger, x_hat.shape[0])

    #     return self.make_results_dict(
    #         stage,
    #         batch,
    #         loss,
    #         reconstruction_loss,
    #         kld_loss,
    #         kld_per_part,
    #         z_parts,
    #         z_parts_params,
    #         z_composed,
    #         x_hat,
    #     )

    # def configure_optimizers(self):

    #     get_params = lambda self, cond: list(self.parameters()) if cond else []
    #     _parameters = (
    #         get_params(self.encoder, True)
    #         + get_params(self.decoder, True)
    #     )
    #     optimizers = [self.optimizer(_parameters)]

    #     lr_schedulers = [
    #         (
    #             self.lr_scheduler(optimizer=optimizers[0])
    #             if self.lr_scheduler is not None
    #             else None
    #         )
    #     ]

    #     self.latent_loss_optimizer_map = dict()
    #     for optim_ix, (part, latent_optim) in enumerate(
    #         self.latent_loss_optimizer.items()
    #     ):
    #         self.latent_loss_optimizer_map[optim_ix] = part
    #         # import ipdb
    #         # ipdb.set_trace()
    #         optimizers.append(
    #             latent_optim(
    #                 self.latent_loss[part].parameters(),
    #             )
    #         )
    #         lr_sched = self.latent_loss_scheduler.get(part)
    #         if lr_sched is not None:
    #             lr_sched = lr_sched(optimizer=optimizers[-1])
    #         lr_schedulers.append(lr_sched)

    #     return optimizers, lr_schedulers

    def _step(self, stage, batch, batch_idx, logger):
        (
            x_hat,
            z_parts,
            z_parts_params,
            z_composed,
            loss,
            reconstruction_loss,
            kld_loss,
            kld_per_part,
        ) = self.forward(batch, decode=True, compute_loss=True)
        if stage != 'train':
            print(reconstruction_loss, z_composed.max(), batch['drug_dose'].unique())
            # import ipdb
            # ipdb.set_trace()
        _loss = {}
        for part in self.latent_loss.keys():
            if isinstance(self.latent_loss[part].loss, torch.nn.modules.loss.BCEWithLogitsLoss):
                batch[self.latent_loss_target[part]] = batch[self.latent_loss_target[part]].gt(0).float()

            _loss[part] = self.latent_loss[part](
                z_parts_params[self.hparams.x_label], batch[self.latent_loss_target[part]]
            ) * self.latent_loss_weights.get(part, 1.0) 

        # import ipdb
        # ipdb.set_trace()
        optimizers = self.optimizers()
        lr_schedulers = self.lr_schedulers()
        main_optim = optimizers.pop(0)
        main_lr_sched = lr_schedulers.pop(0)

        opt_adv = optimizers[0]
        opt_dos = optimizers[1]

        if (batch_idx % 3 == 0) & (stage == 'train'):

            adv_loss = _loss['drug'] + _loss['cell_type']
            opt_adv.zero_grad()
        
            self.manual_backward(
                adv_loss
            )
            opt_adv.step()
        elif stage == 'train':
            main_optim.zero_grad()
            opt_dos.zero_grad()
            (
                reconstruction_loss
                - _loss['drug'] 
                - _loss['cell_type']
            ).backward()
            main_optim.step()
            opt_dos.step()

        self.log_metrics(stage, reconstruction_loss, kld_loss, loss, logger, x_hat.shape[0])

        return self.make_results_dict(
            stage,
            batch,
            loss,
            reconstruction_loss,
            kld_loss,
            kld_per_part,
            z_parts,
            z_parts_params,
            z_composed,
            x_hat,
        )

    def configure_optimizers(self):

        get_params = lambda self, cond: list(self.parameters()) if cond else []
        encoders = {i: self.encoder[i] for i in ['X', 'drug', 'cell_type']}

        _parameters = (
            get_params(nn.ModuleDict(encoders), True)
            + get_params(self.decoder, True)
        )
        optimizers = [self.optimizer(_parameters)]
        lr_schedulers = []

        # lr_schedulers = [
        #     (
        #         self.lr_scheduler(optimizer=optimizers[0])
        #         if self.lr_scheduler is not None
        #         else None
        #     )
        # ]

        _parameters2 = (
            get_params(self.latent_loss['drug'], True)
            + get_params(self.latent_loss['cell_type'], True)
        )

        optimizers.append(
            self.optimizer(
                _parameters2
            )
        )
        optimizers.append(self.optimizer(
            self.encoder['dose'].parameters()
        ))

        # Now schedulers
        lr_schedulers.append(torch.optim.lr_scheduler.StepLR(
            optimizers[0], step_size=45
        ))

        lr_schedulers.append(torch.optim.lr_scheduler.StepLR(
            optimizers[1], step_size=45
        ))

        lr_schedulers.append(torch.optim.lr_scheduler.StepLR(
            optimizers[2], step_size=45
        ))
        # import ipdb
        # ipdb.set_trace()
        return optimizers, lr_schedulers

    # def configure_optimizers(self):

    #     optimizers = []
    #     schedulers = []

    #     get_params = lambda self, cond: list(self.parameters()) if cond else []
    #     _parameters = (
    #         get_params(self.encoder[self.hparams.x_label], True)
    #         + get_params(self.decoder, True)
    #         + get_params(self.encoder[self.continuous_labels[1]], True)
    #     )

    #     for label in self.discrete_labels:
    #         _parameters.extend(get_params(self.encoder[label], True))

    #     optimizers.append(self.latent_loss_optimizer[self.hparams.x_label](
    #         _parameters,
    #     ))

    #     optimizers.append(self.latent_loss_optimizer[self.continuous_labels[1]](
    #         self.encoder[self.continuous_labels[0]].parameters(),
    #     ))

    #     _parameters = get_params(self.latent_loss[self.continuous_labels[1]], True)
    #     for label in self.discrete_labels:
    #         _parameters.extend(get_params(self.latent_loss[label], True))

    #     optimizers.append(self.latent_loss_optimizer[self.discrete_labels](
    #         _parameters,
    #     ))

    #     for index, (key, sch) in enumerate(self.latent_loss_scheduler.items()):
    #         schedulers.append(sch(optimizers[index]))
    #         # schedulers.append(sch)

    #     return optimizers, schedulers
