import torch


def diagonal_gaussian_kl(
    mu1: torch.Tensor,
    mu2: torch.Tensor,
    logvar1: torch.Tensor,
    logvar2: torch.Tensor,
):
    """Computes the Kullback-Leibler divergence between two diagonal gaussians (not
    necessarily isotropic). It also works batch-wise.

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

    return 0.5 * (
        (logvar2 - logvar1)
        + (logvar1 - logvar2).exp()
        + (mu_diff.pow(2) / logvar2.exp())
        + -1
    )


def isotropic_gaussian_kl(mean, log_var):
    """Computes the Kullback-Leibler divergence between a diagonal gaussian and an
    isotropic (0,1) gaussian. It also works batch-wise.

    Parameters
    ----------
    mu: torch.Tensor
        Mean of the gaussian (or batch of gaussians)

    logvar: torch.Tensor
        Log-variance of the gaussian (or batch of gaussians)
    """

    return -0.5 * (1 + log_var - mean.pow(2) - log_var.exp())
