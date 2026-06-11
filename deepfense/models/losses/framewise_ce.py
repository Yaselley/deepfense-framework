"""Framewise cross-entropy for dense temporal labels."""

import torch
import torch.nn as nn

from deepfense.models.base_model import BaseLoss
from deepfense.utils.registry import register_loss


@register_loss("FramewiseCrossEntropy")
class FramewiseCrossEntropy(BaseLoss):
    """Per-frame cross-entropy on ``(B, T, D)`` embeddings and ``(B, T)`` targets."""

    def __init__(self, config):
        super().__init__(config)
        self.in_dim = config["embedding_dim"]
        self.num_classes = config["n_classes"]
        self.ignore_index = int(config.get("ignore_index", -100))

        self.fc = nn.Linear(self.in_dim, self.num_classes)

        class_weights = config.get("class_weights", None)
        weight = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=self.ignore_index,
            reduction=config.get("reduction", "mean"),
        )

    def forward(self, embeddings, targets, logits=None):
        if embeddings.dim() != 3:
            raise ValueError(f"FramewiseCrossEntropy expects (B,T,D), got {tuple(embeddings.shape)}")
        if logits is None:
            logits = self.get_logits(embeddings)

        logits, targets = self._align(logits, targets)

        b, t, c = logits.shape
        flat_targets = targets.reshape(b * t)
        return self.criterion(logits.reshape(b * t, c), flat_targets)

    def _align(self, logits, targets):
        t_log = logits.size(1)
        t_tgt = targets.size(1)
        if t_log == t_tgt:
            return logits, targets
        if t_tgt < t_log:
            pad = t_log - t_tgt
            targets = nn.functional.pad(targets, (0, pad), value=self.ignore_index)
            return logits, targets
        return logits, targets[:, :t_log]

    def get_logits(self, embeddings):
        return self.fc(embeddings)

    def get_score(self, embeddings):
        logits = self.get_logits(embeddings)
        if self.num_classes == 2:
            return logits[..., self.bonafide_label] - logits[..., self.spoof_label]
        return logits
