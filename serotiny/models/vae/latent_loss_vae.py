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
        # if stage == 'train':
        #     print(reconstruction_loss, z_composed.max(), batch['drug_dose'].unique())

        _loss = {}
        for part in self.latent_loss.keys():
            if isinstance(self.latent_loss[part].loss, torch.nn.modules.loss.BCEWithLogitsLoss):
                batch[self.latent_loss_target[part]] = batch[self.latent_loss_target[part]].gt(0).float()

            _loss[part] = self.latent_loss[part](
                z_parts_params[self.hparams.x_label], batch[self.latent_loss_target[part]]
            ) * self.latent_loss_weights.get(part, 1.0) 

        optimizers = self.optimizers()
        lr_schedulers = self.lr_schedulers()
        main_optim = optimizers.pop(0)
        main_lr_sched = lr_schedulers.pop(0)

        non_main_key = [i for i in self.optimizer.keys() if i != 'main']
        non_main_optims = []
        non_main_lr_scheds = []
        for i in non_main_key:
            non_main_optims.append(optimizers.pop(0))
            non_main_lr_scheds.append(lr_schedulers.pop(0))

        adversarial_flag = False
        for optim_ix, (optim, lr_sched) in enumerate(
            zip(optimizers, lr_schedulers)
        ):

            group_key = self.latent_loss_optimizer_map[optim_ix]
            parts = self.latent_loss_optimizer[group_key]['keys']
            mod = self.latent_loss_backprop_when.get(group_key) or 3
            adv_loss = 0
            if stage == 'train':
                for part in parts:  
                    adv_loss += _loss[part]
                if batch_idx % mod == 0:
                    adversarial_flag = True
                    optim.zero_grad()
                    self.manual_backward(adv_loss)
                    optim.step()
                    # Dont use LR scheduler here, messes up the loss
                    # if lr_sched is not None:
                    #     lr_sched.step()

                    on_step = stage == "train"

                    self.log(
                        f"{stage}_{part}_latent_loss",
                        adv_loss.detach().cpu(),
                        logger=logger,
                        on_step=on_step,
                        on_epoch=True,
                    )

        if (stage == "train") & (adversarial_flag == False):
            main_optim.zero_grad()
            for non_main_optim in non_main_optims:
                non_main_optim.zero_grad()
            self.manual_backward(loss - adv_loss)
            main_optim.step()
            for non_main_optim in non_main_optims:
                non_main_optim.step()
            # Dont use LR scheduler here, messes up the loss
            # if main_lr_sched is not None:
            #     main_lr_sched.step()
            # for non_main_lr_sched in non_main_lr_scheds:
            #     non_main_lr_sched.step()

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

        get_params = lambda self: list(self.parameters())

        _parameters = get_params(self.decoder)
        for part in self.optimizer['main']['keys']:
            _parameters.extend(get_params(self.encoder[part]))

        optimizers = [self.optimizer['main']['opt'](_parameters)]

        lr_schedulers = [
            (
                self.lr_scheduler['main'](optimizer=optimizers[0])
                if self.lr_scheduler['main'] is not None
                else None
            )
        ]

        non_main_key = [i for i in self.optimizer.keys() if i != 'main'][0]
        if len(non_main_key) > 0:
            _parameters2 = get_params(self.encoder[self.optimizer[non_main_key]['keys'][0]])
            if len(self.optimizer[non_main_key]['keys']) > 1:
                for key in self.optimizer[non_main_key]['keys'][1:]:
                    _parameters2.extend(get_params(self.encoder[self.optimizer[non_main_key]['keys'][key]]))
            optimizers.append(self.optimizer[non_main_key]['opt'](_parameters2))
            lr_schedulers.append(self.lr_scheduler[non_main_key](optimizer=optimizers[-1]))

        self.latent_loss_optimizer_map = dict()
        for optim_ix, (group_key, group) in enumerate(self.latent_loss_optimizer.items()):
            self.latent_loss_optimizer_map[optim_ix] = group_key
            _parameters3 = (
                get_params(self.latent_loss[group['keys'][0]])
            )
            if len(group['keys']) > 1:
                for key in group['keys'][1:]:
                    _parameters3.extend(get_params(self.latent_loss[key]))
            optimizers.append(group['opt'](_parameters3))
            lr_schedulers.append(self.latent_loss_scheduler[group_key](optimizer=optimizers[-1]))

        return optimizers, lr_schedulers
