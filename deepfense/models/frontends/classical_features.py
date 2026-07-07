"""CQCC / MFCC / LFCC extraction (ported from DeepFense/Multi_View/feature_extract.py)."""

from __future__ import annotations

from typing import Iterable

import numpy as np

try:
    from librosa.feature import delta
    from spafe.features.cqcc import cqcc
    from spafe.features.lfcc import lfcc
    from spafe.features.mfcc import mfcc
except ImportError as exc:  # pragma: no cover - optional runtime dependency
    raise ImportError(
        "Classical feature extraction requires spafe and librosa. "
        "Install with: pip install spafe librosa"
    ) from exc

SUPPORTED_SPECTRAL_TYPES = (
    "mfcc",
    "lfcc",
    "cqcc",
    "mfcc+lfcc",
    "mfcc+cqcc",
    "cqcc+lfcc",
    "mfcc+lfcc+cqcc",
)

# 20 static + 20 delta + 20 delta-delta per cepstral stream.
_SINGLE_STREAM_DIM = 60
SPECTRAL_FEATURE_DIMS: dict[str, int] = {
    "mfcc": _SINGLE_STREAM_DIM,
    "lfcc": _SINGLE_STREAM_DIM,
    "cqcc": _SINGLE_STREAM_DIM,
    "mfcc+lfcc": 120,
    "mfcc+cqcc": 120,
    "cqcc+lfcc": 120,
    "mfcc+lfcc+cqcc": 180,
}


def resolve_spectral_dim(feature_type: str) -> int:
    if feature_type not in SPECTRAL_FEATURE_DIMS:
        raise ValueError(
            f"Unknown spectral feature type {feature_type!r}. "
            f"Choose from {SUPPORTED_SPECTRAL_TYPES}"
        )
    return SPECTRAL_FEATURE_DIMS[feature_type]


def _extract_single_stream(
    audio: np.ndarray,
    *,
    feature_type: str,
    sr: int,
    num_ceps: int,
    nfilts: int,
    cqcc_bins: int,
    compute_delta: bool,
    compute_delta_delta: bool,
) -> np.ndarray:
    if feature_type == "cqcc":
        feats = cqcc(audio, fs=sr, num_ceps=num_ceps, number_of_bins_per_octave=cqcc_bins)
    elif feature_type == "mfcc":
        feats = mfcc(audio, fs=sr, num_ceps=num_ceps, nfilts=nfilts)
    elif feature_type == "lfcc":
        feats = lfcc(audio, fs=sr, num_ceps=num_ceps, nfilts=nfilts)
    else:
        raise ValueError(f"Unsupported single-stream feature: {feature_type}")

    parts = [feats]
    if compute_delta:
        parts.append(delta(feats, order=1))
    if compute_delta_delta:
        parts.append(delta(feats, order=2))
    return np.concatenate(parts, axis=1)


def extract_classical_features(
    audio: np.ndarray,
    feature_type: str = "mfcc",
    *,
    sr: int = 16000,
    num_ceps: int = 20,
    nfilts: int = 70,
    cqcc_bins: int = 96,
    compute_delta: bool = True,
    compute_delta_delta: bool = True,
) -> np.ndarray:
    """
    Extract classical cepstral features from a 1-D waveform.

    Returns:
        np.ndarray of shape ``(T, F)`` where ``F`` is 60, 120, or 180 depending
        on ``feature_type``.
    """
    if feature_type not in SUPPORTED_SPECTRAL_TYPES:
        raise ValueError(
            f"feature_type must be one of {SUPPORTED_SPECTRAL_TYPES}, got {feature_type!r}"
        )

    streams = feature_type.split("+")
    if len(streams) == 1:
        return _extract_single_stream(
            audio,
            feature_type=streams[0],
            sr=sr,
            num_ceps=num_ceps,
            nfilts=nfilts,
            cqcc_bins=cqcc_bins,
            compute_delta=compute_delta,
            compute_delta_delta=compute_delta_delta,
        )

    chunks = [
        _extract_single_stream(
            audio,
            feature_type=stream,
            sr=sr,
            num_ceps=num_ceps,
            nfilts=nfilts,
            cqcc_bins=cqcc_bins,
            compute_delta=compute_delta,
            compute_delta_delta=compute_delta_delta,
        )
        for stream in streams
    ]
    min_len = min(chunk.shape[0] for chunk in chunks)
    aligned = [chunk[:min_len] for chunk in chunks]
    return np.concatenate(aligned, axis=1)


def extract_classical_features_batch(
    waveforms: Iterable[np.ndarray],
    feature_type: str,
    **kwargs,
) -> list[np.ndarray]:
    """Extract classical features for a list of 1-D waveforms."""
    return [extract_classical_features(waveform, feature_type, **kwargs) for waveform in waveforms]
