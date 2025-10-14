import torch
import torch.nn as nn
import torchaudio
import numpy as np
from utils import append_deltas

# ====================================================
# LFCC Frontend
# ====================================================

class LFCCFrontend(nn.Module):
    def __init__(
        self,
        sample_rate=16000,
        n_lfcc=40,
        n_filter=70,
        n_fft=512,
        hop_length=160,
        use_delta=False,
        use_delta_delta=False,
    ):
        super().__init__()
        self.lfcc = torchaudio.transforms.LFCC(
            sample_rate=sample_rate,
            n_lfcc=n_lfcc,
            speckwargs={
                "n_fft": n_fft,
                "hop_length": hop_length,
                "win_length": n_fft,
            },
            n_filter=n_filter,
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
        lfcc = self.lfcc(x)  # (B, n_lfcc, T)
        lfcc = append_deltas(lfcc, self.use_delta, self.use_delta_delta)
        return lfcc.permute(0, 2, 1)
