import torch
import torch.nn as nn
import numpy as np

from .abstract_prior import Prior


class IsotropicGaussianPrior(Prior):
    def __init__(self, indices=None):
        super().__init__(indices)

    @classmethod
    def kl_divergence(cls, mean, logvar, reduction="sum"):
        """Computes the Kullback-Leibler divergence between a diagonal gaussian
        and an isotropic (0,1) gaussian. It also works batch-wise.

        Parameters
        ----------
        mean: torch.Tensor
            Mean of the gaussian (or batch of gaussians)

        logvar: torch.Tensor
            Log-variance of the gaussian (or batch of gaussians)
        """

        kl = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())

        if reduction == "none":
            return kl
        elif reduction == "mean":
            return kl.mean(dim=-1)
        else:
            return kl.sum(dim=-1)

    @classmethod
    def sample(cls, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mean)

    def forward(self, z, mode="kl", **kwargs):
        mean_logvar = z[:, self.indices]
        mean, logvar = torch.split(mean_logvar, mean_logvar.shape[1] // 2, dim=1)

        if mode == "kl":
            return self.kl_divergence(mean, logvar, **kwargs)
        else:
            return self.sample(mean, logvar, **kwargs)


class DiagonalGaussianPrior(IsotropicGaussianPrior):
    def __init__(
        self,
        indices=None,
        mean=None,
        logvar=None,
        learn_mean=False,
        learn_logvar=False,
        dimension=None,
    ):
        super().__init__(indices)

        if None in [logvar, mean]:
            if dimension is not None:
                self.dimension = dimension
            elif hasattr(indices, "__len__"):
                self.dimension = len(indices)
            else:
                raise ValueError(
                    "If you don't specify `indices` you must ", "specify `dimension`"
                )

        if logvar is None:
            logvar = torch.zeros(self.dimension)
        else:
            logvar = torch.tensor(np.fromiter(logvar, dtype=float))

        if learn_logvar:
            logvar = nn.Parameter(logvar, requires_grad=True)

        self.logvar = logvar

        if mean is None:
            mean = torch.zeros(self.dimension)
        else:
            mean = torch.tensor(np.fromiter(mean, dtype=float))

        if learn_mean:
            mean = nn.Parameter(mean, requires_grad=True)

        self.mean = mean

    @classmethod
    def kl_divergence(cls, mu1, mu2, logvar1, logvar2, reduction="sum"):
        """Computes the Kullback-Leibler divergence between two diagonal
        gaussians (not necessarily isotropic). It also works batch-wise.

        Parameters
        ----------
        mu1: torch.Tensor
            Mean of the first gaussian (or batch of first gaussians)

        mu2: torch.Tensor
            Mean of the second gaussian (or batch of second gaussians)

        logvar1: torch.Tensor
            Log-variance of the first gaussian (or batch of first gaussians)

        logvar2: torch.Tensor
            Log-variance of the second gaussian (or batch of second gaussians)
        """
        mu_diff = mu2 - mu1

        kl = 0.5 * (
            (logvar2 - logvar1)
            + (logvar1 - logvar2).exp()
            + (mu_diff.pow(2) / logvar2.exp())
            + -1
        )

        if reduction == "none":
            return kl
        elif reduction == "mean":
            return kl.mean(dim=-1)
        else:
            return kl.sum(dim=-1)

    def forward(self, z, mode="kl", **kwargs):
        mean_logvar = z[:, self.indices]
        mean, logvar = torch.split(mean_logvar, mean_logvar.shape[1] // 2, dim=1)

        if mode == "kl":
            return self.kl_divergence(mean, self.mean, logvar, self.logvar, **kwargs)
        else:
            return self.sample(mean, logvar, **kwargs)
