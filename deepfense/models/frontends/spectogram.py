import torch
import torch.nn as nn
import torchaudio
import numpy as np
from utils import append_deltas

# ====================================================
# Spectrogram Frontend
# ====================================================

class SpectrogramFrontend(nn.Module):
    def __init__(
        self,
        n_fft=512,
        hop_length=160,
        win_length=400,
        power=2.0,
        use_delta=False,
        use_delta_delta=False,
    ):
        super().__init__()
        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=n_fft, hop_length=hop_length, win_length=win_length, power=power
        )
        self.use_delta = use_delta
        self.use_delta_delta = use_delta_delta

    def forward(self, x):
        """
        Args:
            x: Tensor (B, L)
        Returns:
            Tensor (B, T, F)
        """
        spec = self.spectrogram(x)  # (B, F, T)
        spec = append_deltas(spec, self.use_delta, self.use_delta_delta)
        return spec.permute(0, 2, 1)  # (B, T, F)
