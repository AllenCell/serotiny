import torch
import numpy as np
import pandas as pd
from serotiny.loss_formulations import calculate_elbo
import torchvision
from serotiny.utils.model_utils import index_to_onehot
from serotiny.metrics.inception import InceptionV3
from serotiny.metrics.calculate_fid import get_activations
from serotiny.metrics.calculate_fid import calculate_fid
from sklearn.decomposition import PCA


def get_singular_values(df, cols, n_principal_components=None):
    df = df[cols]
    pca = PCA(n_principal_components).fit(df)
    return pca.singular_values_


def visualize_encoder_tabular(
    model,
    conds,
    X_test,
    C_test,
    datamodule,
    value=1,
    beta=1,
    resample_n=10,
    mask=True,
    kl_per_lt=None,
    kl_vs_rcl=None,
):
    """
    Run a test set through the model to get embeddings

    Parameters
    ----------

    model: a trained model set in eval() mode

    conds: Number of possible conditions - usually
    conds = [i for i in enc_layers[0]], i.e. give all of input data
    as a condition or not

    X_test: The input X to the model. Pass in a batch

    C_test: The input C (condition) to the model. Pass in a batch

    conds: In the case of Gaussian datamodule, this specifies
    which columns in condition to set to 0
    In the case of Spharm datamodule, this specifies
    which integer to provide as condition

    beta: What beta to use to compute the loss

    resample_n: How many times to sample z (default = 10)

    mask: Whether to not compute loss when there is missing data or not.
    Default True

    kl_per_lt, kl_vs_rcl: a dataframe that is produced by this function.

    """
    z_means_x, z_means_y = [], []
    z_var_x, z_var_y = [], []

    # switch off gradients
    with torch.no_grad():

        # Make empty dicts to append to
        if kl_per_lt is None:
            kl_per_lt = {
                "latent_dim": [],
                "kl_divergence": [],
                "condition": [],
            }

        if kl_vs_rcl is None:
            kl_vs_rcl = {"condition": [], "KLD": [], "RCL": [], "ELBO": []}
        all_kl, all_lt = [], []

        # Split condition into tmp1 that contains the condition, and
        # tmp2 that contains the mask info of whether the condition
        # is there or not
        if (
            isinstance(conds, list)
            and datamodule.__module__ == "serotiny.datamodules.gaussian"
        ):
            tmp1, tmp2 = torch.split(C_test, int(C_test.size()[-1] / 2), dim=1)

            # conds can be, for example [0,1,2]
            # so set those columns to 0
            # if all set to 0, that means no condition is provided
            for kk in conds:
                tmp1[:, kk], tmp2[:, kk] = 0, 0

            # New condition tensor for the model
            cond_d = torch.cat((tmp1, tmp2), 1)
        elif (
            isinstance(conds, list)
            and datamodule.__module__ == "serotiny.datamodules.variance_spharm_coeffs"
        ):
            cond_d = torch.zeros(C_test.shape)
            cond_d[:, conds] = value

        # Make empty list
        my_recon_list, my_z_means_list, my_log_var_list = [], [], []

        # Run resample_n times for resampling
        for resample in range(resample_n):
            recon_batch, z_means, log_var, _, _, _, _, _ = model(
                X_test.clone().type_as(C_test), cond_d.clone().type_as(C_test)
            )
            my_recon_list.append(recon_batch)
            my_z_means_list.append(z_means)
            my_log_var_list.append(log_var)

        # Average over the N resamples
        recon_batch = torch.mean(torch.stack(my_recon_list), dim=0)
        z_means = torch.mean(torch.stack(my_z_means_list), dim=0)
        log_var = torch.mean(torch.stack(my_log_var_list), dim=0)

        # Calculate loss for this
        (
            elbo_loss_total,
            rcl_per_lt_temp_total,
            kl_per_lt_temp_total,
            _,
            _,
        ) = calculate_elbo(
            X_test.type_as(C_test),
            recon_batch.type_as(C_test),
            z_means,
            log_var,
            beta,
            mask,
        )

        elbo_loss_total = elbo_loss_total / X_test.shape[0]
        rcl_per_lt_temp_total = rcl_per_lt_temp_total / X_test.shape[0]
        kl_per_lt_temp_total = kl_per_lt_temp_total / X_test.shape[0]

        # Save info to dataframe
        # if conds = [0,1,2] for a 2D Gaussian (X_test.size()[-1]),
        # then num_conds = 0, so
        # num_conds = X_test.size()[-1] - len(conds)
        kl_vs_rcl["condition"].append(str(conds))
        kl_vs_rcl["KLD"].append(kl_per_lt_temp_total.item())
        kl_vs_rcl["RCL"].append(rcl_per_lt_temp_total.item())
        kl_vs_rcl["ELBO"].append(elbo_loss_total.item())

        # Calculate loss per latent dimension (z_means.size()[-1])
        for ii in range(z_means.size()[-1]):
            elbo_loss, rcl_per_lt_temp, kl_per_lt_temp, _, _ = calculate_elbo(
                X_test.type_as(C_test),
                recon_batch.type_as(C_test),
                z_means[:, ii],
                log_var[:, ii],
                beta,
                mask,
            )

            # Save all_kl and all_lt, useful for sorting later
            all_kl = np.append(all_kl, kl_per_lt_temp.item())
            all_lt.append(ii)
            kl_per_lt["condition"].append(str(conds))
            kl_per_lt["latent_dim"].append(ii)
            kl_per_lt["kl_divergence"].append(kl_per_lt_temp.item())

        # Sort both kl and lt dim together
        all_kl, all_lt = list(zip(*sorted(zip(all_kl, all_lt))))

        # Convert to list
        all_kl = list(all_kl)
        all_lt = list(all_lt)

        # Save the z_means and z_var for the 2 most important latent dims
        # These are sorted in ascending order, so most imp is -1
        z_means_x = np.append(z_means_x, z_means[:, all_lt[-1]].data.cpu().numpy())
        z_means_y = np.append(z_means_y, z_means[:, all_lt[-2]].data.cpu().numpy())
        z_var_x = np.append(z_var_x, log_var[:, all_lt[-1]].data.cpu().numpy())
        z_var_y = np.append(z_var_y, log_var[:, all_lt[-2]].data.cpu().numpy())

        # return
    return z_means_x, z_means_y, kl_per_lt, z_var_x, z_var_y, kl_vs_rcl


def get_sorted_klds(latent_dict):
    df = pd.DataFrame(latent_dict)
    df = df.sort_values(by=["kl_divergence"])
    n_dim = np.max(df["latent_dim"])

    kld_avg_dim = np.zeros(n_dim)

    for i in range(n_dim):
        kld_avg_dim[i] = np.mean(df["kl_divergence"][df["latent_dim"] == i])
    kld_avg_dim = np.sort(kld_avg_dim)[::-1]

    return kld_avg_dim


def compute_generative_metric_tabular(X_test, C_test, gpu_id, enc_layers, model, conds):

    with torch.no_grad():
        device = (
            torch.device("cuda", gpu_id)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        latent_dims = enc_layers[-1]

        x = X_test.to(device)

        y = C_test.to(device)

        z = torch.randn(X_test.size()[0], latent_dims).to(device)

        generated_x = model.decoder(z, y)

        all_elements = [i for i in range(enc_layers[0])]
        inverse_conds = [i for i in all_elements if i not in conds]
        for kk in inverse_conds:
            generated_x[:, int(kk)] = x[:, int(kk)]

        X_act_mu = np.mean(x.cpu().numpy(), axis=0)
        recon_act_mu = np.mean(generated_x.cpu().numpy(), axis=0)
        X_act_sigma = np.cov(x.cpu().numpy(), rowvar=False)
        recon_act_sigma = np.cov(generated_x.cpu().numpy(), rowvar=False)

        fid = calculate_fid(
            X_act_mu, X_act_sigma, recon_act_mu, recon_act_sigma, eps=1e-6
        )

        return fid


def compute_generative_metric(
    test_iterator,
    model,
    device,
    LATENT_DIM,
    BATCH_SIZE,
    color_value=None,
    digit_value=None,
):
    inc = InceptionV3([3])
    inc = inc.cuda()

    with torch.no_grad():
        im = torch.empty([0])
        lab = torch.empty([0])
        for imm, tll in iter(test_iterator):
            for tim, tl in zip(imm, tll):
                if lab.size()[0] != 500:
                    if digit_value is not None:
                        if tl == digit_value:
                            im = torch.cat((im, tim), 0)
                            lab = torch.cat((lab, tl.view(1).float()), 0)
                    else:
                        im = torch.cat((im, tim), 0)
                        lab = torch.cat((lab, tl.view(1).float()), 0)
                elif lab.size()[0] == 500:
                    break
        im = im.view(lab.size()[0], 1, 28, 28)
        im = im.repeat(1, 3, 1, 1)

        colors = []

        for j in range(lab.size()[0]):
            if color_value is not None:
                color = torch.randint(color_value + 1, color_value + 2, (1, 1)).item()
            else:
                color = torch.randint(1, 4, (1, 1)).item()
            other_indices = []
            # color_index = []
            for a in [1, 2, 3]:
                if color != a:
                    other_indices.append(a)
                else:
                    other_index = a
            im[j, other_indices[0] - 1, :, :].fill_(0)
            im[j, other_indices[1] - 1, :, :].fill_(0)
            colors.append(color - 1)

        colors = torch.FloatTensor(colors)

        z = torch.randn(lab.size()[0], LATENT_DIM).to(device)

        if digit_value is not None:
            y = torch.randint(digit_value, digit_value + 1, (lab.size()[0], 1)).to(
                dtype=torch.long
            )
        else:
            y = torch.randint(0, 10, (lab.size()[0], 1)).to(dtype=torch.long)

        y = index_to_onehot(y, n=10).to(device, dtype=z.dtype)

        y = torch.cat((y, torch.zeros([lab.size()[0]]).view(-1, 1).cuda()), dim=1)

        if color_value is not None:
            y2 = torch.randint(color_value, color_value + 1, (lab.size()[0], 1)).to(
                dtype=torch.long
            )
        else:
            y2 = torch.randint(0, 3, (lab.size()[0], 1)).to(dtype=torch.long)

        y2 = index_to_onehot(y2, n=3).to(device, dtype=z.dtype)
        y2 = torch.cat((y2, torch.zeros([lab.size()[0]]).view(-1, 1).cuda()), dim=1)
        y = torch.cat((y, y2), dim=1)

        z = torch.cat((z, y), dim=1)

        generated_x = model.decoder(z, y)

        X_act = get_activations(
            im.cpu().data.numpy(), inc, batch_size=BATCH_SIZE, dims=2048, cuda=True
        )
        recon_act = get_activations(
            generated_x.cpu().data.numpy(),
            inc,
            batch_size=BATCH_SIZE,
            dims=2048,
            cuda=True,
        )

        X_act_mu = np.mean(X_act, axis=0)
        recon_act_mu = np.mean(recon_act, axis=0)
        X_act_sigma = np.cov(X_act, rowvar=False)
        recon_act_sigma = np.cov(recon_act, rowvar=False)

        fid = calculate_fid(
            X_act_mu, X_act_sigma, recon_act_mu, recon_act_sigma, eps=1e-6
        )

        images = im[:5, :, :, :]
        gen_images = generated_x[:5, :, :, :]

        grid = torchvision.utils.make_grid(images, nrow=5)
        grid2 = torchvision.utils.make_grid(gen_images, nrow=5)

    return fid, grid.cpu(), grid2.cpu()
