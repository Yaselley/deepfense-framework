import torch
import torch.nn as nn
from deepfense.utils.registry import register_loss


@register_loss("CrossEntropy")
class CrossEntropy(nn.Module):
    """
    Unified CrossEntropy Loss + Linear Projection.
    """

    def __init__(self, config):
        super().__init__()
        self.in_dim = config["embedding_dim"]
        self.num_classes = config["n_classes"]

        # Mapper part
        self.fc = nn.Linear(self.in_dim, self.num_classes)

        # Loss part
        class_weights = config.get("class_weights", None)
        reduction = config.get("reduction", "mean")
        
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float)
            
        self.criterion = nn.CrossEntropyLoss(weight=class_weights, reduction=reduction)

    def forward(self, embeddings, targets, logits=None):
        """
        Args:
            embeddings: Tensor of shape (batch, embedding_dim)
            targets: LongTensor of shape (batch,) with class indices
            logits: Optional pre-computed logits to avoid re-calculation.
        """
        if logits is None:
            logits = self.fc(embeddings)
        return self.criterion(logits, targets)

    def get_logits(self, embeddings):
        """Returns logits for validation/inference."""
        return self.fc(embeddings)
