import numpy as np
import soundfile as sf
import librosa

from deepfense.data.transforms.registry import register_transform

@register_transform("load_audio")
def load_audio(
        path: str, 
        target_sr: int = 16000, 
        mono: bool = True
    ):
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
        max_len: int = 64600, 
        random_pad: bool = False, 
        pad_type: str = "repeat"  # "repeat" or "zero"
    ):
    """
    Pad or truncate a waveform to a fixed length.

    Args:
        x (np.ndarray): Input waveform, shape (L,) or (L, 1)
        max_len (int): Target length
        random_pad (bool): If True, randomly select start when truncating
        pad_type (str): "repeat" to repeat waveform, "zero" to zero-pad

    Returns:
        np.ndarray: Padded or truncated waveform
    """
    x_len = x.shape[0]

    # Truncate if longer than max_len
    if x_len > max_len:
        if random_pad:
            start = np.random.randint(0, x_len - max_len)
            return x[start:start + max_len]
        else:
            return x[:max_len]

    # Pad if shorter than max_len
    pad_len = max_len - x_len
    if pad_type == "repeat":
        repeats = int(np.ceil(max_len / x_len))
        padded = np.tile(x, repeats)[:max_len]
    elif pad_type == "zero":
        padded = np.zeros(max_len, dtype=x.dtype)
        padded[:x_len] = x
    else:
        raise ValueError(f"Unknown pad_type: {pad_type}. Use 'repeat' or 'zero'.")

    return padded
