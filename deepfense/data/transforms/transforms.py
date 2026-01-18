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
    max_len: int = 64000,
    random_pad: bool = False,
    pad_type: str = "repeat",  # "repeat"
):
    """
    Pad or truncate a waveform to a fixed length.

    Args:
        x (np.ndarray): Input waveform, shape (L,) or (L, 1)
        max_len (int): Target length
        random_pad (bool): If True, randomly select start when truncating
        pad_type (str): "repeat" to repeat waveform

    Returns:
        np.ndarray: Padded or truncated waveform
    """
    x_len = x.shape[0]

    # Truncate if longer than max_len
    if x_len > max_len:
        if random_pad:
            start = np.random.randint(0, x_len - max_len)
            return x[start : start + max_len]
        else:
            return x[:max_len]

    # Pad if shorter than max_len
    if pad_type == "repeat":
        repeats = int(np.ceil(max_len / x_len))
        padded = np.tile(x, repeats)[:max_len]
    else:
        raise ValueError(
            f"Unknown pad_type: {pad_type}. Only 'repeat' is supported for now."
        )

    return padded
