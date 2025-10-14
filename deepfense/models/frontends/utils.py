import torch
import torch.nn as nn
import torchaudio
import numpy as np


def append_deltas(x: torch.Tensor, use_delta=False, use_delta_delta=False):
    """
    Appends delta and delta-delta features along the feature dimension.
    Args:
        x: Tensor (B, F, T)
        use_delta: bool
        use_delta_delta: bool
    Returns:
        Tensor (B, F*(1+delta+delta_delta), T)
    """
    feats = [x]
    if use_delta:
        delta = torchaudio.functional.compute_deltas(x)
        feats.append(delta)
    if use_delta_delta:
        delta2 = torchaudio.functional.compute_deltas(
            torchaudio.functional.compute_deltas(x)
        )
        feats.append(delta2)
    return torch.cat(feats, dim=1)
