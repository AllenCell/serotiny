import torch
import torch.nn as nn

from .abstract_prior import Prior


class JointPrior(Prior):
    """Prior class that doesn't contribute to KL loss.

    Effectively a Dirac delta distribution given z.
    """

    def __init__(self, priors):
        dimensionality = sum(prior.dimensionality for prior in priors)
        super().__init__(dimensionality)

        if not isinstance(priors, nn.ModuleList):
            priors = nn.ModuleList(*priors)
        self.priors = priors

    def kl_divergence(self, z_params, reduction="sum"):
        kl = []

        start_ix = 0
        for prior in self.priors:
            end_ix = (start_ix + prior.dimensionality) - 1
            kl.append(
                prior(z_params[:, start_ix:end_ix], mode="kl", reduction=reduction)
            )
            start_ix = end_ix + 1

        if reduction == "none":
            return torch.cat(kl, axis=1)

        return torch.sum(kl)

    def sample(self, z_params):
        samples = []

        start_ix = 0
        for prior in self.priors:
            end_ix = (start_ix + prior.dimensionality) - 1
            samples.append(prior(z_params[:, start_ix:end_ix], mode="sample"))
            start_ix = end_ix + 1

        return torch.cat(samples, axis=1)

    def forward(self, z_params, mode="kl", **kwargs):
        if mode == "kl":
            return self.kl_divergence(z_params, **kwargs)
        else:
            return self.sample(z_params, **kwargs)
