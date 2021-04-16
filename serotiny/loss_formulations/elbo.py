import torch


def calculate_elbo(x, reconstructed_x, mean, log_var, beta, mask=False):
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
    kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    kld_per_element = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp())
    print(rcl)
    print(rcl_per_element)

    return rcl + beta * kld, rcl, kld, rcl_per_element, kld_per_element
