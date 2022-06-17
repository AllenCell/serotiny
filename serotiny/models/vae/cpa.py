from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from serotiny.networks.mlp import MLP
from torch.nn.modules.loss import _Loss as Loss
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from serotiny.models.base_model import BaseModel

Array = Union[torch.Tensor, np.array, Sequence[float]]


class CPA(BaseModel):
    def __init__(
        self,
        x_dim: int,
        latent_dim: int,
        hidden_layers: Sequence[int],
        adversary_hidden_layers: Sequence[int],
        x_label: str,
        c_continuous_label: str,
        c_discrete_labels: list,
        id_label: Optional[str] = None,
        beta: float = 1.0,
        prior_mode: str = "isotropic",
        prior_logvar: Optional[Array] = None,
        learn_prior_logvar: bool = False,
        num_continuous_conditions: int=5,
        num_discrete_conditions: list = [1],
        cache_outputs: Sequence = ("test",),
        autoencoder_lr: float = 3e-4,
        autoencoder_wd: float = 4e-7,
        dosers_lr: float = 4e-3,
        dosers_wd: float = 1e-7,
        reg_adversary: int = 60,
        # penalty_adversary: int = 60,
        penalty_adversary: int = 0,
        adversary_lr: float = 3e-4,
        adversary_wd: float = 4e-7,
        adversary_steps: int = 3,
        step_size_lr: int = 45,
        reconstruction_loss: Loss = nn.GaussianNLLLoss(),
        reconstruction_reduce: str = "sum",
        log_transform: Optional[bool] = False,
        doser_type: str = 'logsigm',
    ):

        encoder = MLP(
            x_dim,
            latent_dim,
            hidden_layers=hidden_layers,
        )

        decoder = MLP(
            latent_dim,
            x_dim*2,
            hidden_layers=hidden_layers,
        )

        adversary_cont_conds = MLP(
            latent_dim,
            num_continuous_conditions,
            hidden_layers=adversary_hidden_layers,
        )

        cont_conds_embeddings = torch.nn.Embedding(
            num_continuous_conditions, 
            latent_dim
        )

        dosers = GeneralizedSigmoid(num_continuous_conditions, nonlin=doser_type)

        if num_discrete_conditions == [0]:
            pass
        else:
            assert 0 not in num_discrete_conditions
            adversary_disc_conds = nn.ModuleList()
            loss_adversary_disc_conds = []
            disc_conds_embeddings = (
                []
            )  
            for num_discrete_condition in num_discrete_conditions:
                adversary_disc_conds.append(
                    MLP(
                        latent_dim,
                        num_discrete_condition,
                        hidden_layers=adversary_hidden_layers
                    )
                )
                loss_adversary_disc_conds.append(torch.nn.CrossEntropyLoss())
                disc_conds_embeddings.append(
                    torch.nn.Embedding(num_discrete_condition, latent_dim)
                )
            disc_conds_embeddings = torch.nn.Sequential(
                *disc_conds_embeddings
            )

        super().__init__(
            encoder=encoder,
            decoder=decoder,
            latent_dim=latent_dim,
            x_label=x_label,
            id_label=id_label,
            beta=beta,
            prior_mode=prior_mode,
            prior_logvar=prior_logvar,
            learn_prior_logvar=learn_prior_logvar,
            cache_outputs=cache_outputs,
            reconstruction_loss=reconstruction_loss,
            reconstruction_reduce=reconstruction_reduce,
        )
        self.x_label = x_label
        self.num_cont_conds = num_continuous_conditions
        self.num_disc_conds = num_discrete_conditions
        self.doser_type = doser_type
        self.dosers = dosers
        self.adversary_cont_conds = adversary_cont_conds
        self.loss_adversary_cont_conds = torch.nn.BCEWithLogitsLoss() 
        self.disc_conds_embeddings = disc_conds_embeddings
        self.loss_adversary_disc_conds = loss_adversary_disc_conds
        self.adversary_disc_conds = adversary_disc_conds
        self.cont_conds_embeddings = cont_conds_embeddings
        self.c_discrete_labels = c_discrete_labels
        self.c_continuous_label = c_continuous_label
        self.autoencoder_wd = autoencoder_wd
        self.autoencoder_lr = autoencoder_lr
        self.adversary_lr = adversary_lr
        self.adversary_wd = adversary_wd
        self.adversary_steps = adversary_steps
        self.step_size_lr = step_size_lr
        self.reg_adversary = reg_adversary
        self.dosers_lr = dosers_lr
        self.dosers_wd = dosers_wd
        self.log_transform = log_transform
        self.penalty_adversary = penalty_adversary
        # manual gradients
        self.automatic_optimization = False

        self.reconstruction_loss = reconstruction_loss
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.latent_dim = latent_dim

    def parse_batch(self, batch):
        discrete_batches = []
        for i in self.c_discrete_labels:
            if len(batch[i].shape) != 2:
                discrete_batches.append(batch[i].view(-1,1))
            else:
                discrete_batches.append(batch[i])
        return batch[self.x_label].float(), discrete_batches, batch[self.c_continuous_label].float()

    def compute_cont_conds_embeddings_(self, cont_conds):
        """
        """
        return self.dosers(cont_conds) @ self.cont_conds_embeddings.weight

    def forward(self, x, **kwargs):

        inputs, disc_conds, cont_conds = x[0],x[1],x[2]
        # if self.log_transform:
        #     inputs = torch.log1p(inputs)

        latent_basal = self.encoder(inputs)
        latent_perturbed = latent_basal.float()

        if self.num_cont_conds > 0:
            # import ipdb
            # ipdb.set_trace()
            latent_perturbed = latent_perturbed + self.compute_cont_conds_embeddings_(cont_conds).float()
        if self.num_disc_conds[0] > 0:
            for i, emb in enumerate(self.disc_conds_embeddings):
                latent_perturbed = latent_perturbed + emb(
                    disc_conds[i].argmax(1)
                ).float()  #argmax because OHE

        reconstructions = self.decoder(latent_perturbed.float())

        if isinstance(self.reconstruction_loss, nn.GaussianNLLLoss):
            dim = reconstructions.size(1) // 2
            recon_means = reconstructions[:, :dim]
            recon_vars = F.softplus(reconstructions[:, dim:]).add(1e-3)

        reconstructions = torch.cat([recon_means, recon_vars], dim=1)

        return (
            latent_basal,
            latent_perturbed,
            reconstructions,
            recon_means,
            recon_vars,
        )

    def _step(self, stage, batch, batch_idx, logger):

        x = self.parse_batch(batch)
        inputs, disc_conds, cont_conds = x[0],x[1],x[2]

        (
            latent_basal,
            latent_perturbed,
            reconstructions,
            recon_means,
            recon_vars,
        ) = self.forward(x)

        reconstruction_loss = self.reconstruction_loss(recon_means, inputs, recon_vars)
        if stage == 'train':
            print(reconstruction_loss, batch['drug_dose'].unique())
        adversary_cont_conds_loss = torch.tensor([0.0]).type_as(inputs)

        if self.num_cont_conds > 0:
            adversary_cont_conds_predictions = self.adversary_cont_conds(latent_basal)
            adversary_cont_conds_loss = self.loss_adversary_cont_conds(
                adversary_cont_conds_predictions, cont_conds.gt(0).float()
            )

        adversary_disc_conds_loss = torch.tensor([0.0]).type_as(inputs)
        if self.num_disc_conds[0] > 0:
            adversary_disc_conds_predictions = []
            for i, adv in enumerate(self.adversary_disc_conds):
                adversary_disc_conds_predictions.append(adv(latent_basal))
                adversary_disc_conds_loss += self.loss_adversary_disc_conds[i](
                    adversary_disc_conds_predictions[-1], disc_conds[i].argmax(1)
                )

        # two place-holders for when adversary is not executed
        adversary_cont_conds_penalty = torch.tensor([0.0]).type_as(inputs)
        adversary_disc_conds_penalty = torch.tensor([0.0]).type_as(inputs)

        opt_auto, opt_adv, opt_dos = self.optimizers()
        lr_auto, lr_adv, lr_dos = self.lr_schedulers()

        if (batch_idx % self.adversary_steps == 0) & (stage == 'train'):
            if self.num_cont_conds > 0:
                adversary_cont_conds_penalty = compute_gradients(
                    adversary_cont_conds_predictions.sum(), latent_basal
                )
            if self.num_disc_conds[0] > 0:
                adversary_disc_conds_penalty = torch.tensor([0.0]).type_as(inputs)
                for pred in adversary_disc_conds_predictions:
                    adversary_disc_conds_penalty += compute_gradients(
                        pred.sum(), latent_basal
                    )  
            opt_adv.zero_grad()
        
            self.manual_backward(
                adversary_cont_conds_loss
                + self.penalty_adversary * adversary_cont_conds_penalty
                + adversary_disc_conds_loss
                + self.penalty_adversary * adversary_disc_conds_penalty
            )
            opt_adv.step()
        if (stage == 'train') & (batch_idx % self.adversary_steps != 0):
            opt_auto.zero_grad()
            if self.num_cont_conds > 0:
                opt_dos.zero_grad()
            self.manual_backward(
                reconstruction_loss
                - self.reg_adversary * adversary_cont_conds_loss
                - self.reg_adversary * adversary_disc_conds_loss
            )
            opt_auto.step()
            if self.num_cont_conds > 0:
                opt_dos.step()

        tot_loss = reconstruction_loss - ( adversary_cont_conds_loss + adversary_disc_conds_loss)

        self.log_metrics(stage, tot_loss, reconstruction_loss, 
        adversary_disc_conds_loss, adversary_cont_conds_loss, 
        adversary_disc_conds_penalty, adversary_cont_conds_penalty, logger)

        results = {
            "loss": tot_loss,
            f"{stage}_loss": reconstruction_loss.detach().cpu(),  # for epoch end logging purposes
            "adversary_disc_conds_loss": adversary_disc_conds_loss.detach().cpu(),
            "adversary_cont_conds_loss": adversary_cont_conds_loss.detach().cpu(),
            "adversary_cont_conds_penalty": adversary_cont_conds_penalty.detach().cpu(),
            "adversary_disc_conds_penalty": adversary_disc_conds_penalty.detach().cpu(),
            "batch_idx": batch_idx,
        }

        if stage == "test":
            results.update(
                {
                    "recon_means": recon_means.detach().cpu(),
                    "recon_vars": recon_vars.detach().cpu(),
                    "x": x.detach().cpu(),
                }
            )

        if self.hparams.id_label is not None:
            if self.hparams.id_label in batch:
                ids = batch[self.hparams.id_label].detach().cpu()
                results.update({self.hparams.id_label: ids, "id": ids})

        return results

    def log_metrics(
        self, stage, loss, reconstruction_loss, 
        adversary_disc_conds_loss, adversary_cont_conds_loss, 
        adversary_disc_conds_penalty, adversary_cont_conds_penalty, logger):

        self.log(f"{stage}_loss", loss, on_step=True, logger=logger)
        self.log(f"{stage} reconstruction loss", reconstruction_loss, on_step=True,  logger=logger)
        self.log(f"{stage} adversary_discrete_condition_loss", adversary_disc_conds_loss, on_step=True, logger=logger)
        self.log(f"{stage} adversary_continuous_condition_loss", adversary_cont_conds_loss, on_step=True, logger=logger)
        self.log(f"{stage} adversary_continuous_condition_penalty", adversary_cont_conds_penalty, on_step=True, logger=logger)
        self.log(f"{stage} adversary_discrete_condition_penalty", adversary_disc_conds_penalty, on_step=True, logger=logger)

    def configure_optimizers(self):

        # First optimizers
        has_discrete_conditions = self.num_disc_conds[0] > 0
        has_continuous_conditions = self.num_cont_conds > 0
        get_params = lambda self, cond: list(self.parameters()) if cond else []
        _parameters = (
            get_params(self.encoder, True)
            + get_params(self.decoder, True)
            + get_params(self.cont_conds_embeddings, has_continuous_conditions)
        )

        for emb in self.disc_conds_embeddings:
            _parameters.extend(get_params(emb, has_discrete_conditions))

        self.optimizer_autoencoder = torch.optim.Adam(
            _parameters,
            lr=self.autoencoder_lr,
            weight_decay=self.autoencoder_wd,
        )
        _parameters = get_params(self.adversary_cont_conds, has_continuous_conditions)
        for adv in self.adversary_disc_conds:
            _parameters.extend(get_params(adv, has_discrete_conditions))

        self.optimizer_adversaries = torch.optim.Adam(
            _parameters,
            lr=self.adversary_lr,
            weight_decay=self.adversary_wd,
        )

        if has_continuous_conditions:
            self.optimizer_dosers = torch.optim.Adam(
                self.dosers.parameters(),
                lr=self.dosers_lr,
                weight_decay=self.dosers_wd,
            )

        optimizers = [self.optimizer_autoencoder, self.optimizer_adversaries, self.optimizer_dosers]

        # Now schedulers
        self.scheduler_autoencoder = torch.optim.lr_scheduler.StepLR(
            self.optimizer_autoencoder, step_size=self.step_size_lr
        )

        self.scheduler_adversary = torch.optim.lr_scheduler.StepLR(
            self.optimizer_adversaries, step_size=self.step_size_lr
        )

        if has_continuous_conditions:
            self.scheduler_dosers = torch.optim.lr_scheduler.StepLR(
                self.optimizer_dosers, step_size=self.step_size_lr
            )   
        schedulers = [self.scheduler_autoencoder, self.scheduler_adversary, self.scheduler_dosers]

        return optimizers, schedulers

def compute_gradients(output, input):
    grads = torch.autograd.grad(output, input, create_graph=True)
    grads = grads[0].pow(2).mean()
    return grads

class GeneralizedSigmoid(torch.nn.Module):
    """
    Sigmoid, log-sigmoid or linear functions for encoding dose-response for
    drug perurbations.
    """

    def __init__(self, dim, nonlin="sigmoid"):
        """Sigmoid modeling of continuous variable.
        Params
        ------
        nonlin : str (default: logsigm)
            One of logsigm, sigm.
        """
        super(GeneralizedSigmoid, self).__init__()
        self.nonlin = nonlin
        self.beta = torch.nn.Parameter(
            torch.ones(1, dim), requires_grad=True
        )
        self.bias = torch.nn.Parameter(
            torch.zeros(1, dim), requires_grad=True
        )

    def forward(self, x):
        if self.nonlin == "logsigm":
            # import ipdb
            # ipdb.set_trace()
            # self.bias = self.bias.type_as(x)
            c0 = self.bias.sigmoid()
            return (torch.log1p(x) * self.beta + self.bias).sigmoid() - c0
        elif self.nonlin == "sigm":
            # self.bias = self.bias.type_as(x)
            c0 = self.bias.sigmoid() 
            return (x * self.beta + self.bias).sigmoid() - c0
        else:
            return x

    def one_drug(self, x, i):
        if self.nonlin == "logsigm":
            c0 = self.bias[0][i].sigmoid()
            return (torch.log1p(x) * self.beta[0][i] + self.bias[0][i]).sigmoid() - c0
        elif self.nonlin == "sigm":
            c0 = self.bias[0][i].sigmoid()
            return (x * self.beta[0][i] + self.bias[0][i]).sigmoid() - c0
        else:
            return x

class NBLoss(torch.nn.Module):
    def __init__(self):
        super(NBLoss, self).__init__()

    def forward(self, mu, y, theta, eps=1e-8):
        """Negative binomial negative log-likelihood. It assumes targets `y` with n
        rows and d columns, but estimates `yhat` with n rows and 2d columns.
        The columns 0:d of `yhat` contain estimated means, the columns d:2*d of
        `yhat` contain estimated variances. This module assumes that the
        estimated mean and inverse dispersion are positive---for numerical
        stability, it is recommended that the minimum estimated variance is
        greater than a small number (1e-3).
        Parameters
        ----------
        yhat: Tensor
                Torch Tensor of reeconstructed data.
        y: Tensor
                Torch Tensor of ground truth data.
        eps: Float
                numerical stability constant.
        """
        if theta.ndimension() == 1:
            # In this case, we reshape theta for broadcasting
            theta = theta.view(1, theta.size(0))
        log_theta_mu_eps = torch.log(theta + mu + eps)
        res = (
            theta * (torch.log(theta + eps) - log_theta_mu_eps)
            + y * (torch.log(mu + eps) - log_theta_mu_eps)
            + torch.lgamma(y + theta)
            - torch.lgamma(theta)
            - torch.lgamma(y + 1)
        )
        res = _nan2inf(res)
        return -torch.mean(res)