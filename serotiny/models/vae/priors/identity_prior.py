from .abstract_prior import Prior


class IdentityPrior(Prior):
    """Prior class that doesn't contribute to KL loss.

    Effectively a Dirac delta distribution given z.
    """

    def forward(self, z, mode="kl", **kwargs):
        if mode == "kl":
            return 0
        else:
            return z
