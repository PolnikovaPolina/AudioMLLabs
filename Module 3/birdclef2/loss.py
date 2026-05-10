import torchvision
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        return torchvision.ops.focal_loss.sigmoid_focal_loss(
            inputs=inputs,
            targets=targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )
