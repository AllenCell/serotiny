import torch.nn.functional as F
from torch.nn.modules.loss import _Loss as Loss
import torch


class GaussianNLLLoss(Loss):
    def __init__(self, reduction="mean"):
        super(GaussianNLLLoss, self).__init__(None, None, reduction)
        self.reduction = reduction
        self.loss = torch.nn.GaussianNLLLoss(reduction=self.reduction)

    def forward(self, input, target):
        dim = target.size(1) // 2
        recon_means = target[:, :dim]
        recon_vars = F.softplus(target[:, dim:]).add(1e-3)
        target = torch.cat([recon_means, recon_vars], dim=1)
        loss = self.loss(recon_means, input, recon_vars)
        return loss
