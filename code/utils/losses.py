import math
import torch
import torch.nn as nn


def soft_dice_multiclass(probs, targets_onehot, eps=1e-6, ignore_bg=False):
    inter = (probs * targets_onehot).sum(dim=(0, 2, 3))
    denom = (probs + targets_onehot).sum(dim=(0, 2, 3)) + eps
    dice_k = (2.0 * inter + eps) / denom
    if ignore_bg:
        return 1.0 - dice_k[1:].mean()
    return 1.0 - dice_k.mean()


class CEDiceLoss(nn.Module):
    """Cross entropy + soft Dice used in the original PUMix notebook."""

    def __init__(self, ce_weight=0.5, eps=1e-6, ignore_bg=False):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.ce_weight = ce_weight
        self.eps = eps
        self.ignore_bg = ignore_bg

    def forward(self, logits, target):
        ce = self.ce(logits, target.long())
        probs = torch.softmax(logits, dim=1)
        onehot = torch.zeros_like(probs)
        onehot.scatter_(1, target.long().unsqueeze(1), 1.0)
        dice = soft_dice_multiclass(
            probs, onehot, eps=self.eps, ignore_bg=self.ignore_bg
        )
        return self.ce_weight * ce + (1.0 - self.ce_weight) * dice


def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    current = max(0.0, min(float(current), float(rampup_length)))
    phase = 1.0 - current / rampup_length
    return float(math.exp(-5.0 * phase * phase))


def consistency_weight(epoch, max_weight=0.2, rampup_epochs=20, base=0.0):
    return base + (max_weight - base) * sigmoid_rampup(epoch, rampup_epochs)
