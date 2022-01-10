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
    recon_loss: Loss = torch.nn.MSELoss,
    mode="isotropic",
    prior_mu=None,
    prior_logvar=None,
    **kwargs,
):

    recon_loss = recon_loss(reduction="none")

    mask_kwargs = {k: v for k, v in kwargs.items() if k in ['mask']}
    if len(mask_kwargs) != 0:
        mask = mask_kwargs['mask']
        reconstructed_x = reconstructed_x[mask]
        x = x[mask]
        mean = mean[mask]
        log_var = log_var[mask]

    rcl_per_input_dimension = recon_loss(reconstructed_x, x)
    rcl = rcl_per_input_dimension.sum(dim=1).sum()
    # rcl = rcl_per_input_dimension.sum(dim=1).mean()

    if mode == "isotropic":
        kld_per_dimension = isotropic_gaussian_kl(mean, log_var)
    elif mode == "anisotropic":
        kld_per_dimension = diagonal_gaussian_kl(mean, prior_mu, log_var, prior_logvar)

    else:
        raise NotImplementedError(f"KLD mode '{mode}' not implemented")

    kld = kld_per_dimension.sum(dim=1).mean()
    # kld = kld_per_dimension.sum(dim=1).sum()

    return (
        rcl + beta * kld,
        rcl,
        kld,
        rcl_per_input_dimension,
        kld_per_dimension,
    )
