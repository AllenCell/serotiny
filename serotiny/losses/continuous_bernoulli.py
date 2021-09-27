from torch.nn.modules.loss import _Loss as Loss

from torch.distributions.continuous_bernoulli import ContinuousBernoulli


class CBLogLoss(Loss):
    """
    Continuous Bernoulli loss, proposed here: https://arxiv.org/abs/1907.06845
    """

    def __init__(self, reduction="sum", mode="probs"):
        super(CBLogLoss, self).__init__(None, None, reduction)
        self.mode = mode

    def forward(self, input, target):

        # the trick with the dictionary allows us to use either `probs` or `logits`
        log_probs = ContinuousBernoulli(**{self.mode: input}).log_prob(target)

        if self.reduction == "none":
            return -log_probs
        elif self.reduction == "sum":
            return -log_probs.view(log_probs.shape[0], -1).sum(axis=1)
        elif self.reduction == "mean":
            return -log_probs.view(log_probs.shape[0], -1).mean(axis=1)
        else:
            raise NotImplementedError(f"Unavailable reduction type: {self.reduction}")
