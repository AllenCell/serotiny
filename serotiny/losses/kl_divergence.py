import torch
from torch.nn.modules.loss import _Loss as Loss


def diagonal_gaussian_kl(mu1, mu2, logvar1, logvar2):
    mu_diff = mu2 - mu1

    return 0.5 * (
        (logvar2 - logvar1)
        + (logvar1 - logvar2).exp()
        + (mu_diff.pow(2) / logvar2.exp())
        + -1
    )


def isotropic_gaussian_kl(mean, log_var):
    return -0.5 * (1 + log_var - mean.pow(2) - log_var.exp())


def calculate_elbo(
    x,
    reconstructed_x,
    mean,
    log_var,
    beta,
    recon_loss: Loss,
    recon_reduce="mean",
    mask=None,
    mode="isotropic",
    prior_mu=None,
    prior_logvar=None,
):

    rcl_per_input_dimension = recon_loss(reconstructed_x, x)
    if mask is not None:
        rcl_per_input_dimension = rcl_per_input_dimension * mask
        normalizer = mask.view(mask.shape[0], -1).sum(dim=1)
    else:
        normalizer = sum(x.shape[1:])

    rcl = (
        rcl_per_input_dimension
        # flatten
        .view(rcl_per_input_dimension.shape[0], -1)
        # and sum per batch element.
        .sum(dim=1)
    )

    if recon_reduce == "mean":
        rcl = rcl / normalizer

    rcl = rcl.mean()

    if mode == "isotropic":
        kld_per_dimension = isotropic_gaussian_kl(mean, log_var)
    elif mode == "anisotropic":
        kld_per_dimension = diagonal_gaussian_kl(mean, prior_mu, log_var, prior_logvar)

    else:
        raise NotImplementedError(f"KLD mode '{mode}' not implemented")

    kld = kld_per_dimension.sum(dim=1).mean()

    return (
        rcl + beta * kld,
        rcl,
        kld,
        rcl_per_input_dimension,
        kld_per_dimension,
    )
