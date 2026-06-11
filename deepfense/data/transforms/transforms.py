from __future__ import annotations

import numpy as np
import soundfile as sf
import librosa
import os
import logging

from deepfense.utils.registry import register_transform

logger = logging.getLogger(__name__)


@register_transform("load_audio")
def load_audio(path: str, target_sr: int = 16000, mono: bool = True):
    # Check if file exists
    if not os.path.exists(path):
        error_msg = (
            f"Audio file not found: {path}\n"
            f"Please check that the file exists and the path is correct."
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Check if path is a file (not a directory)
    if not os.path.isfile(path):
        error_msg = (
            f"Path is not a file: {path}\n"
            f"Please provide a valid audio file path."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Read the audio file
    x, sr = sf.read(path, always_2d=False)

    # Convert to mono if needed
    if mono and x.ndim > 1:
        x = np.mean(x, axis=1)

    # Resample if needed
    if sr != target_sr:
        x = librosa.resample(x, orig_sr=sr, target_sr=target_sr)

    return x


@register_transform("pad")
def pad_combined(
    x: np.ndarray,
    max_len: int | None = 64000,
    random_pad: bool = False,
    pad_type: str = "repeat",
    truncate: bool = True,
):
    """
    Optionally pad / truncate a waveform.

    Args:
        x: Input waveform, shape ``(L,)`` or ``(L, 1)`` (first dim is time).
        max_len: Target length for padding shorter clips when set. If ``None``,
            the waveform is returned unchanged (full clip; rely on batch collate).
        random_pad: If True and ``truncate`` is True, random crop when truncating.
        pad_type: ``"repeat"`` (tile), ``"zeros"`` / ``"zero"`` / ``"constant"`` (zero-fill).
        truncate: If True (default), trim inputs longer than ``max_len``.
            Set False for **full-audio / partial-deepfake** training so the file
            is never cropped in the dataset (variable length; pad in collate).

    Returns:
        Padded/truncated array, or the original waveform if ``max_len`` is ``None``.
    """
    x = np.asarray(x)
    x_len = int(x.shape[0])

    if max_len is None:
        return x

    # Truncate if longer than max_len
    if x_len > max_len:
        if not truncate:
            return x
        if random_pad:
            start = np.random.randint(0, x_len - max_len)
            return x[start : start + max_len]
        return x[:max_len]

    if x_len == max_len:
        return x

    if pad_type in ("repeat",):
        repeats = int(np.ceil(max_len / max(x_len, 1)))
        padded = np.tile(x, repeats)[:max_len]
    elif pad_type in ("zeros", "zero", "constant"):
        padded = np.zeros((max_len,), dtype=x.dtype)
        padded[:x_len] = x
    else:
        raise ValueError(
            f"Unknown pad_type: {pad_type}. Use 'repeat' or 'zeros'."
        )

    return padded
