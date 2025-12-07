import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple

class BaseFrontend(nn.Module, ABC):
    """
    Base class for all Frontends.
    Frontends take raw audio waveform and return features.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Raw audio waveform [Batch, Time]
        Returns:
            features: [Batch, Time, Channels] or similar
        """
        pass
    
    @property
    def output_dim(self) -> int:
        """Return the dimension of the output features."""
        return self.config.get("output_dim", 0)


class BaseBackend(nn.Module, ABC):
    """
    Base class for all Backends.
    Backends take features from frontend and return embeddings.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Features from frontend
        Returns:
            embeddings: [Batch, Embedding_Dim]
        """
        pass

    @property
    def input_dim(self) -> int:
        """Return the expected input dimension."""
        return self.config.get("input_dim", 1024)


class BaseLoss(nn.Module, ABC):
    """
    Base class for all Unified Loss Modules (Mapper + Loss).
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        # Standardize binary classification labels
        self.bonafide_label = config.get("bonafide_label", 1)
        # Assuming binary classification (0 vs 1)
        self.spoof_label = abs(1 - self.bonafide_label)

    @abstractmethod
    def forward(self, embeddings: torch.Tensor, targets: torch.Tensor, logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the loss.
        Args:
            embeddings: [Batch, Emb_Dim]
            targets: [Batch]
            logits: Optional pre-computed logits
        Returns:
            loss: scalar tensor
        """
        pass

    @abstractmethod
    def get_score(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute scores for evaluation (e.g. LLR).
        Args:
            embeddings: [Batch, Emb_Dim]
        Returns:
            scores: [Batch] or [Batch, N_Classes]
        """
        pass
    
    def get_logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Returns raw logits/cosines for caching/loss computation.
        Defaults to get_score if not overridden, but subclasses 
        should implement this to return full [N, C] output.
        """
        return self.get_score(embeddings)

