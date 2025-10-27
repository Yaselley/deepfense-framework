import torch
import torch.nn as nn
from deepfense.training.losses.registry import register_loss

@register_loss("CrossEntropy")
class CrossEntropy(nn.Module):
    """
    Wrapper around torch.nn.CrossEntropyLoss with optional class weights.
    """

    def __init__(self, class_weights=None, reduction='mean'):
        """
        Args:
            class_weights: list or tensor of weights for each class (optional)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights, reduction=reduction)

    def forward(self, logits, targets):
        """
        Args:
            logits: Tensor of shape (batch, num_classes)
            targets: LongTensor of shape (batch,) with class indices
        """
        return self.criterion(logits, targets)
