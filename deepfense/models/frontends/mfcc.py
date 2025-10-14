import torch
import torch.nn as nn
import torchaudio
import numpy as np
from utils import append_deltas

# ====================================================
# MFCC Frontend
# ====================================================

class MFCCFrontend(nn.Module):
    def __init__(
        self,
        sample_rate=16000,
        n_mfcc=40,
        n_mels=64,
        n_fft=512,
        hop_length=160,
        use_delta=False,
        use_delta_delta=False,
    ):
        super().__init__()
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": n_fft,
                "n_mels": n_mels,
                "hop_length": hop_length,
                "mel_scale": "htk",
            },
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
        mfcc = self.mfcc(x)  # (B, n_mfcc, T)
        mfcc = append_deltas(mfcc, self.use_delta, self.use_delta_delta)
        return mfcc.permute(0, 2, 1)

