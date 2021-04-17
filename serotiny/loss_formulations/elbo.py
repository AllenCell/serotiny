import torch


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
    loss_per_element = torch.nn.MSELoss(size_average=False, reduce=False)

    # if x is masked, find indices in x
    # that are 0 and set reconstructed x to 0 as well
    if mask:
        indices = x == 0
        reconstructed_x[indices] = 0

    rcl = loss(reconstructed_x, x)
    rcl_per_element = loss_per_element(reconstructed_x, x)

    if mode == "isotropic":
        kld_per_element = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp())
    elif mode == "anisotropic":
        mu_diff = prior_mu - mean

        kld_per_element = 0.5 * (
            (prior_logvar - log_var) +
            (log_var - prior_logvar).exp() +
            (mu_diff.pow(2) / prior_logvar.exp()) +
            -1
        )
    else:
        raise NotImplementedError(f"KLD mode '{mode}' not implemented")

    kld = kld_per_element.sum()

    return rcl + beta * kld, rcl, kld, rcl_per_element, kld_per_element
