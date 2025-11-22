import os
import sys
import copy
import math
import numpy as np
import logging
from pathlib import Path
import pandas as pd
import soundfile as sf
import torchaudio
import librosa
import random
from typing import Optional, List
import torch
import torch.nn.functional as F
from scipy import signal
from pathlib import Path
from deepfense.data.transforms.registry import register_transform
from deepfense.data.transforms.RawBoost.data_utils_rawboost import (
    process_Rawboost_feature,
    get_default_args,
)

logger = logging.getLogger(__name__)

# start: some helper functions

def select_audio(csv_file: str, 
                 sample_rate: int = 16000) -> np.ndarray:
    """
    csv_file path should have a list of paths
    the function randomly read a path from the csv file
    load and return the audio
    """
    df = pd.read_csv(csv_file)
    path = df.sample(1)["path"].values[0]
    audio, sr = librosa.load(path, sr=sample_rate, mono=True)
    return audio


def select_multiple_audio(csv_file: str, 
                          count: int, 
                          sample_rate: int = 16000) -> List[np.ndarray]:
    """
    csv_file path should have a list of paths
    the function randomly reads count paths from the csv file
    loads and returns a list of audio 
    """
    df = pd.read_csv(csv_file)
    replace = False
    if len(df) < count:
        logger.warning(f"Less than {count} audio files in the csv file, using all of them")
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
                      length: Optional[int], 
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

# End of helper functions

@register_transform("simple_aug")
def sample_aug_func(x, noise_ratio):
    return x


# TODO: We might need try except to catch the exceptions in case
# augmentation failed (e.g., failed to load the rir due to some network issue)
@register_transform("rir")
def rir(x, noise_ratio: float, csv_file: str, sample_rate: int = 16000) -> np.ndarray:
    """
    apply rir augmentation to monochannel audio
    - rir_path: directory containing .wav RIR files
    """
    if np.random.random() > noise_ratio:
        return audio
    audio = x
    rir_audio = select_audio(csv_file, sample_rate)

    audio_power = float((audio**2).mean())
    if audio_power < 1e-10:
        return audio

    # one path should be read randmoly from the csv file
    # rir, sample_rate = audio_util.get_audio(rir_path)  # assume mono, no trim

    augmented = signal.convolve(audio, rir_audio, mode="full")[: audio.shape[0]]

    augment_power = float((augmented**2).mean())
    if augment_power > 1e-10:
        scale = float(np.sqrt(audio_power / augment_power))
        augmented = scale * augmented

    return augmented


@register_transform("rawboost")
def rawboost(x, noise_ratio, param2, param3) -> np.ndarray:
    """
    apply RawBoost augmentation to mono audio
    """

    algo = 5

    audio = x

    # if np.random.random() > noise_ratio:
    #     return audio

    sample_rate = 16000

    parameters = get_default_args()

    try:
        return process_Rawboost_feature(
            feature=audio,
            sr=sample_rate,
            args=parameters,
            algo=algo,
        )
    except Exception as e:
        print(f"Warning: RawBoost augmentation failed, returning original audio: {e}")
        return audio


@register_transform("codec")
def codec(x: dict, noise_ratio: float) -> np.ndarray:
    """
    Apply codec augmentation on mono audio.
    Implementation adapted from speechbrain.
    """
    if np.random.random() > noise_ratio:
        return audio

    audio: np.ndarray = x["audio"]
    sample_rate: int = 16000
    # formats = [("wav", "pcm_mulaw"), ("mp3", None), ("g722", None)]
    # TODO: add more formats
    formats = [("wav", "pcm_mulaw"), ("g722", None)]
    fmt, enc = random.choice(formats)

    x = torch.as_tensor(audio, dtype=torch.float32)
    x = x.unsqueeze(0).transpose(0, 1).cpu()

    eff = torchaudio.io.AudioEffector(format=fmt, encoder=enc)
    y = eff.apply(x, sample_rate).transpose(0, 1).squeeze(0)

    out = y.numpy()
    if np.issubdtype(audio.dtype, np.floating):
        out = out.astype(audio.dtype, copy=False)
    return out


@register_transform("morph")
def morph(
    x: dict,
    noise_ratio: float,
) -> np.ndarray:
    """
    Apply morphing augmentation on mono audio.
    Implementation adapted from ESPNet.
    """
    if np.random.rand() > noise_ratio:
        return audio
    audio: np.ndarray = x["audio"]
    noise: np.ndarray = x["noise"]
    noise_db_low: float = (5,)
    noise_db_high: float = (20,)

    audio_power = float((audio**2).mean())
    noise_db = np.random.uniform(noise_db_low, noise_db_high)

    audio_nsamples = audio.shape[0]
    noise_nsamples = noise.shape[0]

    # align noise and audio such that they have the same length
    if audio_nsamples == noise_nsamples:
        pass
    elif audio_nsamples > noise_nsamples:
        offset = np.random.randint(0, audio_nsamples - noise_nsamples)
        noise = np.pad(
            noise,
            (offset, audio_nsamples - noise_nsamples - offset),
            mode="wrap",
        )
    else:
        offset = np.random.randint(0, noise_nsamples - audio_nsamples)
        noise = noise[offset : offset + audio_nsamples]

    noise_power = float((noise**2).mean())
    scale = (
        10 ** (-noise_db / 20)
        * np.sqrt(audio_power)
        / np.sqrt(np.maximum(noise_power, 1e-10))
    )
    audio = audio + scale * noise
    return audio

@register_transform("add_noise")
def add_noise(audio, 
            noise_ratio, 
            csv_file, 
            snr_low=5,  # use the default values from the ESPNet implementation
            snr_high=20, 
            pad_noise=False, 
            start_index=None, 
            normalize=False, 
            sample_rate=16000):
    """
    Apply noise augmentation to audio, replace the original
    ESPNet implementation.

    """
    # TODO: should we add a sample rate check?

    if np.random.random() > noise_ratio:
        return audio
    
    audio_power = float((audio**2).mean())
    if audio_power < 1e-10:
        logger.warning("Audio power is too low, likely a silent audio, returning original audio")
        return audio
    
    SNR = np.random.uniform(snr_low, snr_high)
    clean_amplitude = compute_amplitude(audio) 
    noise_amplitude_factor = 1 / (dB_to_amplitude(SNR) + 1)
    new_noise_amplitude = noise_amplitude_factor * clean_amplitude
    noisy_waveform = audio * (1 - noise_amplitude_factor)

    noise = select_audio(csv_file, sample_rate)
    audio_len = len(audio)
    noise = align_waveform(noise, audio_len, pad_noise, start_index)
    
    noise_amplitude = compute_amplitude(noise)
    noise_waveform = noise * new_noise_amplitude / (noise_amplitude + 1e-14)
    noisy_waveform += noise_waveform

    return noisy_waveform

@register_transform("speed_perturb")
def speed_perturb(audio, 
            noise_ratio, 
            speeds=[90, 100, 110], 
            perturb_prob=1.0,
            sample_rate=16000):
    """
    Apply speed perturbation augmentation to audio.
    """
    if np.random.random() > noise_ratio:
        return audio
    
    speed = random.choice(speeds)
    new_freq = sample_rate * speed // 100
    resampled = librosa.resample(audio, orig_sr=sample_rate, target_sr=new_freq)
    
    return resampled


@register_transform("add_babble")
def add_babble(audio, 
               noise_ratio, 
               csv_file, 
               speaker_count=3, 
               snr_low=0, 
               snr_high=0, 
               pad_noise=False,
               start_index=None,
               sample_rate=16000):
    if np.random.random() > noise_ratio:
        return audio
    
    babble_waveforms = select_multiple_audio(
        csv_file, 
        speaker_count, 
        sample_rate)
    
    SNR = np.random.uniform(snr_low, snr_high)
    clean_amplitude = compute_amplitude(audio)
    noise_amplitude_factor = 1 / (dB_to_amplitude(SNR) + 1)
    new_noise_amplitude = noise_amplitude_factor * clean_amplitude
    
    babbled_audio = audio * (1 - noise_amplitude_factor)
    
    audio_len = len(audio)
    babble_waveform = np.zeros(audio_len, dtype=audio.dtype)
    for i in range(speaker_count):
        waveform_idx = (1 + i) % speaker_count
        aligned_bw = align_waveform(
            babble_waveforms[waveform_idx], 
            audio_len, pad_noise, start_index)
        babble_waveform += aligned_bw
    
    babble_amplitude = compute_amplitude(babble_waveform)
    babble_waveform *= new_noise_amplitude / (babble_amplitude + 1e-14)
    babbled_audio += babble_waveform
    
    return babbled_audio

@register_transform("drop_freq")
def drop_freq(audio: np.ndarray, 
              noise_ratio: float, 
              drop_freq_low: float = 1e-14, 
              drop_freq_high: float = 1, 
              drop_count_low: int = 1, 
              drop_count_high: int = 2, 
              drop_width: float = 0.05, 
              sample_rate: int = 16000) -> np.ndarray:
    """
    Arguments
    ---------
    drop_freq_low : float
        The low end of frequencies that can be dropped,
        as a fraction of the sampling rate / 2.
    drop_freq_high : float
        The high end of frequencies that can be
        dropped, as a fraction of the sampling rate / 2.
    drop_count_low : int
        The low end of number of frequencies that could be dropped.
    drop_count_high : int
        The high end of number of frequencies that could be dropped.
    drop_width : float
        The width of the frequency band to drop, as
        a fraction of the sampling_rate / 2.
    drop_prob : float
        The probability that the batch of signals will  have a frequency
        dropped. By default, every batch has frequencies dropped.

    Example
    -------
    """
    if np.random.random() > noise_ratio:
        return audio
    
    drop_count = np.random.randint(drop_count_low, drop_count_high + 1)
    drop_range = drop_freq_high - drop_freq_low
    drop_frequencies = np.random.rand(drop_count) * drop_range + drop_freq_low

    # Filter parameters, hard coded just like speechbrain's impl
    filter_length = 101
    pad = filter_length // 2

    # create a delta filter 
    drop_filter = np.zeros(filter_length)
    drop_filter[pad] = 1 # impulse
    
    nyquist = sample_rate / 2
    for frequency in drop_frequencies:
        notch_kernel = notch_filter(frequency, filter_length, drop_width)
        drop_filter = np.convolve(drop_filter, notch_kernel, mode='same')
    
    dropped_audio = np.convolve(audio, drop_filter, mode='same')
    
    return dropped_audio


@register_transform("drop_chunk")
def drop_chunk(audio: np.ndarray, 
               noise_ratio: float, 
               drop_length_low: int = 100, 
               drop_length_high: int = 1000, 
               drop_count_low: int = 1, 
               drop_count_high: int = 10, 
               drop_start: int = 0, 
               drop_end: Optional[int] = None, 
               noise_factor: float = 0.0) -> np.ndarray:
    if np.random.random() > noise_ratio:
        return audio
    
    audio_len = len(audio)
    dropped_audio = audio.copy()
    
    clean_amplitude = compute_amplitude(audio)
    
    drop_times = np.random.randint(drop_count_low, drop_count_high + 1)
    
    for _ in range(drop_times):
        drop_length = np.random.randint(drop_length_low, drop_length_high + 1)
        
        start_min = drop_start
        if start_min < 0:
            start_min += audio_len
        
        start_max = drop_end if drop_end is not None else audio_len
        if start_max < 0:
            start_max += audio_len
        start_max = max(0, start_max - drop_length)
        
        if start_min >= start_max:
            continue
        
        start = np.random.randint(start_min, start_max + 1)
        end = min(start + drop_length, audio_len)
        
        if noise_factor == 0.0:
            dropped_audio[start:end] = 0.0
        else:
            noise_max = 2 * clean_amplitude * noise_factor
            noise_vec = np.random.rand(end - start) * 2 * noise_max - noise_max
            dropped_audio[start:end] = noise_vec
    
    return dropped_audio



@register_transform("do_clip")
def do_clip(audio: np.ndarray, 
            noise_ratio: float, 
            clip_low: float = 0.5, 
            clip_high: float = 1, 
            clip_prob: float = 1) -> np.ndarray:
    if np.random.random() > noise_ratio:
        return audio
    
    clipping_range = clip_high - clip_low
    clip_value = np.random.rand() * clipping_range + clip_low
    clipped_audio = np.clip(audio, -clip_value, clip_value)
    
    return clipped_audio
