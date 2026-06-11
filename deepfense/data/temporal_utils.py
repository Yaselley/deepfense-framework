"""Helpers for aligning dense frame labels with audio and SSL feature rates."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


def approx_num_frames(n_audio_samples: int, hop: int) -> int:
    """Number of frames at hop ``hop`` (floor division, drops trailing partial window)."""
    if n_audio_samples <= 0 or hop <= 0:
        return 0
    return n_audio_samples // hop


def dense_labels_to_fixed(
    labels: Sequence[int] | np.ndarray,
    target_len: int,
    pad_value: int = -100,
) -> np.ndarray:
    """Crop or pad a 1D label vector to ``target_len``."""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    out = np.full((target_len,), pad_value, dtype=np.int64)
    if labels.size == 0:
        return out
    copy_len = min(target_len, labels.size)
    out[:copy_len] = labels[:copy_len]
    return out


def mask_trailing_frames(
    frame_labels: np.ndarray,
    valid_frames: int,
    pad_value: int = -100,
) -> np.ndarray:
    """Set frames ``>= valid_frames`` to ``pad_value`` (ignored in loss)."""
    out = frame_labels.copy()
    if valid_frames < out.shape[0]:
        out[valid_frames:] = pad_value
    return out


_VALID_LABEL_MERGE_RULES = ("any_spoof", "all_spoof", "majority", "any_non_bonafide")


def _majority_class(labels: np.ndarray) -> int:
    """Return the most frequent label; ties go to the smallest class id."""
    uniq, counts = np.unique(labels, return_counts=True)
    return int(uniq[counts.argmax()])


def downsample_frame_labels(
    labels: np.ndarray,
    factor: int,
    rule: str = "any_spoof",
    spoof_label: int = 0,
    bonafide_label: int = 1,
    ignore_value: int = -100,
) -> np.ndarray:
    """Downsample labels by ``factor`` using a partial-deepfake merge rule.

    Binary rules (``any_spoof``, ``all_spoof``, ``majority``): spoof vs bonafide only.

    ``any_non_bonafide`` (multiclass): if any non-bonafide frame appears in a
    window, bonafide loses and the window label is the majority vote among the
    non-bonafide frames; if all valid frames are bonafide, the window is bonafide.
    """
    if rule not in _VALID_LABEL_MERGE_RULES:
        raise ValueError(
            f"Unknown label merge rule '{rule}'. Use one of {_VALID_LABEL_MERGE_RULES}."
        )
    if factor <= 0:
        raise ValueError(f"factor must be a positive integer, got {factor}")

    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if factor == 1:
        return labels.copy()

    n_full = labels.size // factor
    if n_full == 0:
        return np.zeros((0,), dtype=np.int64)

    windows = labels[: n_full * factor].reshape(n_full, factor)
    valid = windows != ignore_value
    has_any_valid = valid.any(axis=1)
    out = np.full((n_full,), ignore_value, dtype=np.int64)

    if rule == "any_spoof":
        is_spoof = (windows == spoof_label) & valid
        out[has_any_valid] = np.where(
            is_spoof.any(axis=1)[has_any_valid], spoof_label, bonafide_label
        )
    elif rule == "all_spoof":
        is_spoof = (windows == spoof_label) & valid
        all_spoof_when_valid = (is_spoof.sum(axis=1) == valid.sum(axis=1))
        out[has_any_valid] = np.where(
            all_spoof_when_valid[has_any_valid], spoof_label, bonafide_label
        )
    elif rule == "majority":
        spoof_counts = ((windows == spoof_label) & valid).sum(axis=1)
        valid_counts = valid.sum(axis=1)
        bonafide_counts = valid_counts - spoof_counts
        out[has_any_valid] = np.where(
            spoof_counts[has_any_valid] >= bonafide_counts[has_any_valid],
            spoof_label,
            bonafide_label,
        )
    elif rule == "any_non_bonafide":
        for i in np.where(has_any_valid)[0]:
            w = windows[i, valid[i]]
            if np.all(w == bonafide_label):
                out[i] = bonafide_label
            else:
                non_bf = w[w != bonafide_label]
                out[i] = _majority_class(non_bf)

    return out


def downsample_mask_to_frames(
    audio_mask: torch.Tensor,
    hop: int,
    target_frames: int | None = None,
) -> torch.Tensor:
    """Convert sample-level mask ``(B, T_audio)`` to frame mask at hop ``hop``."""
    if audio_mask.ndim != 2:
        raise ValueError(
            f"downsample_mask_to_frames expects (B, T_audio); got {tuple(audio_mask.shape)}"
        )
    b, t = audio_mask.shape
    n_full = t // hop
    if n_full <= 0:
        out = audio_mask.new_zeros((b, 0))
    else:
        m = audio_mask[:, : n_full * hop].reshape(b, n_full, hop)
        out = m.min(dim=-1).values

    if target_frames is not None:
        if out.shape[1] >= target_frames:
            out = out[:, :target_frames]
        else:
            pad = audio_mask.new_zeros((b, target_frames - out.shape[1]))
            out = torch.cat([out, pad], dim=1)
    return out
