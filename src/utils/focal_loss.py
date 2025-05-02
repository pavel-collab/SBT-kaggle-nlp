import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        alpha: Веса для классов (можно поставить приоритет на какой-то класс)
        gamma: Насколько сильно фокусироваться на ошибках (обычно 2.0)
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, labels):
        ce_loss = F.cross_entropy(logits, labels, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)  # вероятность предсказания правильного класса
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss