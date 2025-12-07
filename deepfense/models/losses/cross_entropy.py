import torch
import torch.nn as nn
from deepfense.utils.registry import register_loss


from deepfense.models.base_model import BaseLoss

@register_loss("CrossEntropy")
class CrossEntropy(BaseLoss):
    """
    Unified CrossEntropy Loss + Linear Projection.
    """

    def __init__(self, config):
        super().__init__(config)
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

    def get_score(self, embeddings):
        """
        Returns final scores for validation/inference.
        If n_classes == 2, returns (logit_bonafide - logit_spoof).
        Otherwise returns full logits.
        """
        logits = self.get_logits(embeddings)
        if self.num_classes == 2:
            # Return LLR based on configured labels
            return logits[:, self.bonafide_label]
        return logits

    def get_logits(self, embeddings):
        """Returns full logits [N, C] for caching/loss."""
        return self.fc(embeddings)
