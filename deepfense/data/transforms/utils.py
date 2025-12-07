import pandas as pd
import numpy as np
import librosa
import logging
from typing import List, Optional
import random 

logger = logging.getLogger(__name__)

def select_audio(source, sample_rate: int = 16000) -> np.ndarray:
    """
    Selects a random audio file from a source and loads it.
    
    Args:
        source: Can be a string (path to CSV), a pandas DataFrame, or a list of paths.
        sample_rate: Target sample rate.
    """
    if isinstance(source, str):
        df = pd.read_csv(source)
        path = df.sample(1)["path"].values[0]
    elif isinstance(source, pd.DataFrame):
        path = source.sample(1)["path"].values[0]
    elif isinstance(source, list):
        path = random.choice(source)
    else:
        raise ValueError(f"Unsupported source type: {type(source)}")

    audio, sr = librosa.load(path, sr=sample_rate, mono=True)
    return audio

def select_multiple_audio(source, count: int, sample_rate: int = 16000) -> List[np.ndarray]:
    """
    Selects multiple random audio files.
    
    Args:
        source: Can be a string (path to CSV), a pandas DataFrame, or a list of paths.
    """
    if isinstance(source, str):
        df = pd.read_csv(source)
    elif isinstance(source, pd.DataFrame):
        df = source
    elif isinstance(source, list):
        df = pd.DataFrame({"path": source})
    else:
        raise ValueError(f"Unsupported source type: {type(source)}")

    replace = False
    if len(df) < count:
        logger.warning(f"Less than {count} audio files available, reusing some.")
        replace = True
        
    sampled_df = df.sample(count, replace=replace)
    
    audios = []
    for path in sampled_df["path"].values:
        audio, sr = librosa.load(path, sr=sample_rate, mono=True)
        audios.append(audio)
    
    return audios

def align_waveform(waveform: np.ndarray,
                   target_len: int, 
                   pad_noise=False, 
                   start_index=None) -> np.ndarray:
    """
    Align a waveform to target length.
    If pad_noise, loop the waveform; else pad with 0.
    Then select a random start_index and crop to target_len.
    """
    waveform_len = len(waveform)
    
    if pad_noise:
        while waveform_len < target_len:
            prepend = waveform[:waveform_len]
            waveform = np.concatenate([prepend, waveform])
            waveform_len = len(waveform)
    else:
        if waveform_len < target_len:
            waveform = np.pad(waveform, (0, target_len - waveform_len), mode="constant")
            waveform_len = len(waveform)
    
    if start_index is None:
        max_chop = max(1, waveform_len - target_len)
        start_index = np.random.randint(0, max_chop)
    
    waveform = waveform[start_index:start_index + target_len]
    return waveform

def compute_amplitude(waveform: np.ndarray, 
                      length: Optional[int] = None, 
                      amp_type: str = "avg", 
                      scale: str = "linear") -> float:
    """
    Compute the amplitude of a waveform.
    """
    assert amp_type in ["avg", "rms", "peak"]
    assert scale in ["linear", "dB"]
    
    if amp_type == "avg":
        if length is None:
            out = np.mean(np.abs(waveform))
        else:
            out = np.sum(np.abs(waveform)) / length
    elif amp_type == "rms":
        if length is None:
            out = np.sqrt(np.mean(waveform**2))
        else:
            out = np.sqrt(np.sum(waveform**2) / length)
    elif amp_type == "peak":
        out = np.max(np.abs(waveform))
    
    if scale == "linear":
        return out
    elif scale == "dB":
        return np.clip(20 * np.log10(out), -80, None)

def dB_to_amplitude(dB: float) -> float:
    """
    Convert dB to amplitude.
    """
    return 10 ** (dB / 20)

def notch_filter(notch_freq: float, 
                 filter_width: int = 101, 
                 notch_width: float = 0.05) -> np.ndarray:
    assert 0 < notch_freq <= 1
    assert filter_width % 2 != 0
    pad = filter_width // 2
    inputs = np.arange(filter_width) - pad

    notch_freq += notch_width

    def sinc(x):
        def _sinc(x):
            return np.sin(x) / x
        return np.concatenate([_sinc(x[:pad]), np.ones(1), _sinc(x[pad + 1:])])

    hlpf = sinc(3 * (notch_freq - notch_width) * inputs)
    window = np.blackman(filter_width)
    hlpf *= window
    hlpf /= np.sum(hlpf)

    hhpf = sinc(3 * (notch_freq + notch_width) * inputs)
    hhpf *= window
    hhpf /= -np.sum(hhpf)
    hhpf[pad] += 1

    return hlpf + hhpf

