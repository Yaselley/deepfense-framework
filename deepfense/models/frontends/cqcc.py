import torch
import torch.nn as nn
import torchaudio
import numpy as np
from utils import append_deltas

# ====================================================
# CQCC Frontend
# ====================================================

class CQCCFrontend(nn.Module):
    def __init__(
        self,
        sample_rate=16000,
        n_bins=96,
        bins_per_octave=12,
        hop_length=160,
        n_cqcc=40,
        use_delta=False,
        use_delta_delta=False,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.hop_length = hop_length
        self.n_cqcc = n_cqcc
        self.use_delta = use_delta
        self.use_delta_delta = use_delta_delta

        # Precompute DCT matrix for CQCC
        self.register_buffer(
            "dct_matrix",
            torchaudio.functional.create_dct(self.n_cqcc, self.n_bins, "ortho"),
        )

    def forward(self, x):
        """
        Args:
            x: Tensor (B, L)
        Returns:
            Tensor (B, F, T)
        """
        B = x.shape[0]
        cqcc_list = []
        for i in range(B):
            waveform = x[i]
            C = torchaudio.transforms.CQT(
                sample_rate=self.sample_rate,
                n_bins=self.n_bins,
                bins_per_octave=self.bins_per_octave,
                hop_length=self.hop_length,
            )(waveform)

            C = torch.abs(C)
            C_log = torch.log(C + 1e-6)
            cqcc_feat = torch.matmul(self.dct_matrix, C_log)
            cqcc_list.append(cqcc_feat.unsqueeze(0))

        cqcc = torch.cat(cqcc_list, dim=0)  # (B, n_cqcc, T)
        cqcc = append_deltas(cqcc, self.use_delta, self.use_delta_delta)
        return cqcc.permute(0, 2, 1)

RAND = torch.randn(32, 64600).to("cuda")
spc = CQCCFrontend(use_delta=True, use_delta_delta=True).to("cuda")
out = spc(RAND)
print(out.shape)