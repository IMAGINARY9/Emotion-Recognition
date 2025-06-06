import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Focal Loss for multi-class classification."""
    def __init__(self, gamma=2.0, weight=None, reduction='mean', label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, input, target):
        log_probs = F.log_softmax(input, dim=-1)
        probs = torch.exp(log_probs)
        if self.label_smoothing > 0:
            n_classes = input.size(1)
            smooth_target = torch.zeros_like(input).scatter_(1, target.unsqueeze(1), 1)
            smooth_target = smooth_target * (1 - self.label_smoothing) + self.label_smoothing / n_classes
        else:
            smooth_target = torch.zeros_like(input).scatter_(1, target.unsqueeze(1), 1)
        focal_weight = (1 - probs).pow(self.gamma)
        loss = -smooth_target * focal_weight * log_probs
        if self.weight is not None:
            loss = loss * self.weight.unsqueeze(0)
        loss = loss.sum(dim=1)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
