from .abstract_prior import Prior
import torch


class IdentityPrior(Prior):
    """Prior class that doesn't contribute to KL loss.
    Effectively a Dirac delta distribution given z.
    """

    def forward(self, z, mode="kl", **kwargs):
        if mode == "kl":
            return torch.tensor(0)
        else:
            return z
