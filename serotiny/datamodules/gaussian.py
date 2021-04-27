from typing import Optional
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from torch.distributions import MultivariateNormal, Multinomial
import numpy as np
import multiprocessing as mp


class GaussianDataset(Dataset):
    def __init__(
        self,
        length,
        x_label,
        c_label,
        c_label_ind,
        x_dim,
        shuffle=True,
        corr=False,
        binomial=False,
        bimodal=False,
    ):
        """
        Args:
            length: Length of dataset
            x_label: key for gaussian input
            y_label: key for y indicating mask
            x_dim: indicates input data size
            shuffle:  True sets condition vector in input data
            to 0 for all possible permutations
            corr: True sets dependent input dimensions via a correlation matrix
        """
        self.length = length
        self.x_label = x_label
        self.c_label = c_label
        self.c_label_ind = c_label_ind
        self.BATCH_SIZE = 1
        self.corr = corr
        self.shuffle = shuffle
        self.x_dim = x_dim
        self.binomial = binomial
        self.bimodal = bimodal

        Batches_X, Batches_C, Batches_conds = (
            torch.empty([0]),
            torch.empty([0]),
            torch.empty([0]),
        )

        if self.bimodal:
            all_x_dims = []
            for dims in range(self.x_dim):
                N = int(self.length / 2)
                mu, sigma = -50, 10
                mu2, sigma2 = 50, 5
                X1 = np.random.normal(mu, sigma, N)
                X2 = np.random.normal(mu2, sigma2, self.length - N)
                X_concat = np.concatenate([X1, X2])
                all_x_dims.append(X_concat)
            all_x = np.stack(all_x_dims)
            all_x = np.swapaxes(all_x, 0, 1)
            np.random.shuffle(all_x)
            all_x = torch.tensor(all_x)
        # import ipdb

        # ipdb.set_trace()

        for j, i in enumerate(range(self.length)):
            if self.corr is False:
                if self.binomial:
                    m = Multinomial(20, torch.tensor([1.0] * self.x_dim))
                    X = m.sample((self.BATCH_SIZE,))
                elif self.bimodal:
                    X = all_x[j].view([1, -1])
                else:
                    m = MultivariateNormal(
                        torch.zeros(x_dim),
                        torch.eye(x_dim),
                    )
                    X = m.sample((self.BATCH_SIZE,))
            else:
                if j == 0:
                    corr_matrix = self.random_corr_mat(D=x_dim)
                    corr_matrix = torch.from_numpy(corr_matrix)
                m = MultivariateNormal(torch.zeros(x_dim).float(), corr_matrix.float())

            C = X.clone()

            count = 0
            if self.shuffle is True:
                while count == 0:
                    C_mask = torch.zeros(C.shape).bernoulli_(0.5)
                    count = 1
            else:
                C_mask = torch.zeros(C.shape).bernoulli_(0)

            C[C_mask.byte()] = 0
            C_indicator = C_mask == 0

            C = torch.cat([C.float(), C_indicator.float()], 1)
            X = X.view([1, -1, x_dim])
            C = C.view([1, -1, x_dim * 2])

            # Sum up
            conds = C[:, :, x_dim:].sum(2)

            Batches_X = torch.cat([Batches_X, X], 0)
            Batches_C = torch.cat([Batches_C, C], 0)
            Batches_conds = torch.cat([Batches_conds, conds], 0)
        self._batches_x = Batches_X
        self._batches_c = Batches_C
        self._batches_conds = Batches_conds

    def __len__(self):
        return len(self._batches_x)

    def __getitem__(self, idx):
        """
        Returns a tuple. (X, C, sum(C[mid:end])).
        X is the input,
        C is the condition,
        sum(C[mid:end]) is the sum of the indicators in C.
        It tells us how many of the condition
        columns have been masked
        """
        return {
            self.x_label: self._batches_x[idx].squeeze(0),
            self.c_label: self._batches_c[idx].squeeze(0),
            self.c_label_ind: self._batches_conds[idx].squeeze(0),
        }

    def random_corr_mat(self, D=10, beta=1):
        """Generate random valid correlation matrix of dimension D.
        Smaller beta gives larger off diagonal correlations (beta > 0)."""

        P = np.zeros([D, D])
        S = np.eye(D)

        for k in range(0, D - 1):
            for i in range(k + 1, D):
                P[k, i] = 2 * np.random.beta(beta, beta) - 1
                p = P[k, i]
                for l in reversed(range(k)):
                    p = (
                        p * np.sqrt((1 - P[l, i] ** 2) * (1 - P[l, k] ** 2))
                        + P[l, i] * P[l, k]
                    )
                S[k, i] = S[i, k] = p

        p = np.random.permutation(D)
        for i in range(D):
            S[:, i] = S[p, i]
        for i in range(D):
            S[i, :] = S[i, p]
        return S


def make_dataloader(
    length,
    x_label,
    c_label,
    c_label_ind,
    x_dim,
    batch_size,
    num_workers,
    shuffle,
    corr,
    binomial,
    bimodal,
):
    """
    Instantiate gaussian dataset and return dataloader
    """
    dataset = GaussianDataset(
        length,
        x_label,
        c_label,
        c_label_ind,
        x_dim,
        shuffle=shuffle,
        corr=corr,
        binomial=binomial,
        bimodal=bimodal,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=False,
        drop_last=True,
        num_workers=num_workers,
        multiprocessing_context=mp.get_context("fork"),
    )


class GaussianDataModule(pl.LightningDataModule):
    """
    A pytorch lightning datamodule that handles the logic for
    loading a Gaussian dataset

    Parameters
    -----------
    batch_size: int
        batch size for dataloader

    num_workers: int
        Number of worker processes for dataloader

    x_label: str
        x_label key to retrive image

    y_label: str
        y_label key to retrieve image label

    dims: list
        Dimensions for dummy images

    length: int
        Length of dummy dataset

    channels: list = [],
        Number of channels for dummy images
    """

    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        x_label: str,
        c_label: str,
        c_label_ind: str,
        x_dim: list,
        length: int,
        shuffle: Optional[bool] = False,
        corr: Optional[bool] = False,
        binomial: Optional[bool] = False,
        bimodal: Optional[bool] = False,
        **kwargs
    ):

        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.x_label = x_label
        self.c_label = c_label
        self.c_label_ind = c_label_ind
        self.length = length
        self.x_dim = x_dim

        self.dataloader = make_dataloader(
            length,
            x_label,
            c_label,
            c_label_ind,
            x_dim,
            batch_size,
            num_workers,
            shuffle,
            corr,
            binomial,
            bimodal,
        )

    def train_dataloader(self):
        return self.dataloader

    def val_dataloader(self):
        return self.dataloader

    def test_dataloader(self):
        return self.dataloader
