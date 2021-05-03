import torch

def diagonal_gaussian_kl(mu1, mu2, logvar1, logvar2):
    mu_diff = mu2 - mu1

    return 0.5 * (
        (logvar2 - logvar1) +
        (logvar1 - logvar2).exp() +
        (mu_diff.pow(2) / logvar2.exp()) +
        -1
    )

def isotropic_gaussian_kl(mean, log_var):
    return -0.5 * (1 + log_var - mean.pow(2) - log_var.exp())

def calculate_elbo(
    x,
    reconstructed_x,
    mean,
    log_var,
    beta,
    mask=False,
    mode="isotropic",
    prior_mu=None,
    prior_logvar=None,
):
    """
    MSE loss for reconstruction,
    KLD loss as per VAE.
    Also want to output dimension (element) wise RCL and KLD
    """

    loss = torch.nn.MSELoss(size_average=False)
    loss_per_observation = torch.nn.MSELoss(size_average=False, reduce=False)

    # if x is masked, find indices in x
    # that are 0 and set reconstructed x to 0 as well
    if mask:
        indices = x == 0
        reconstructed_x[indices] = 0

    rcl = loss(reconstructed_x, x)
    rcl_per_input_dimension = loss_per_observation(reconstructed_x, x)

    if mode == "isotropic":
        kld_per_dimension = isotropic_gaussian_kl(mean, log_var)
    elif mode == "anisotropic":
        kld_per_dimension = diagonal_gaussian_kl(mean, prior_mu, log_var, prior_logvar)

    else:
        raise NotImplementedError(f"KLD mode '{mode}' not implemented")

    kld = kld_per_dimension.sum()

    return rcl + beta * kld, rcl, kld, rcl_per_input_dimension, kld_per_dimension
