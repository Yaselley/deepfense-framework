import torch
import torch.nn as nn
from deepfense.models.loss_mappers.registry import register_loss
from deepfense.models.loss_mappers.registry import register_mapper

@register_mapper("CrossEntropyMapper")
class CrossEntropyMapper(nn.Module):
    """Simple linear mapper for standard softmax classification."""
    def __init__(self, config):
        super().__init__()
        in_dim = config["embedding_dim"]
        num_classes = config["n_classes"]

        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)  # logits

@register_loss("CrossEntropy")
class CrossEntropy(nn.Module):
    """
    Wrapper around torch.nn.CrossEntropyLoss with optional class weights.
    """

    def __init__(self, config):
        """
        Args:
            class_weights: list or tensor of weights for each class (optional)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()

        class_weights = config.get("class_weights", [0.5, 0.5])
        reduction = config.get("reduction", "mean")
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
