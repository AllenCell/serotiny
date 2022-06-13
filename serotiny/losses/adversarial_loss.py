from torch.nn.modules.loss import _Loss as Loss


class AdversarialLoss(Loss):
    def __init__(self, discriminator, loss, reduction="mean"):
        super(AdversarialLoss, self).__init__(None, None, reduction)
        self.discriminator = discriminator
        self.loss = loss

    def forward(self, input, target):

        yhat = self.discriminator(input)
        loss = self.loss(yhat, target)

        # reduce across batch dimension
        if self.reduction == "none":
            return loss
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "mean":
            return loss.mean()
        else:
            raise NotImplementedError(f"Unavailable reduction type: {self.reduction}")
