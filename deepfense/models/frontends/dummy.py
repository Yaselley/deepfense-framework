"""Lightweight waveform encoder for CPU / smoke tests."""

import torch
import torch.nn as nn

from deepfense.models.base_model import BaseFrontend
from deepfense.utils.registry import register_frontend


@register_frontend("dummy")
class DummyFrontend(BaseFrontend):
    """Strided conv that mimics an SSL hop so backends can run on CPU."""

    def __init__(self, config):
        super().__init__(config)
        self._output_dim = int(config.get("output_dim", 64))
        hop_length = int(config.get("hop_length", 320))
        kernel_size = int(config.get("kernel_size", hop_length))
        self.proj = nn.Conv1d(
            1,
            self._output_dim,
            kernel_size=kernel_size,
            stride=hop_length,
        )

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        feats = self.proj(x.unsqueeze(1)).transpose(1, 2)
        return feats
