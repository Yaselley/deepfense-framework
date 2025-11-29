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
from deepfense.data.transforms.audio_utils import (
    select_audio,
    select_multiple_audio,
    align_waveform,
    compute_amplitude,
    dB_to_amplitude,
    notch_filter,
)

logger = logging.getLogger(__name__)


@register_transform("simple_aug")
class SampleAug:
    def __init__(self, noise_ratio):
        self.noise_ratio = noise_ratio

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        return x


# TODO: We might need try except to catch the exceptions in case
# augmentation failed (e.g., failed to load the rir due to some network issue)
@register_transform("rir")
class RIR:
    def __init__(self, noise_ratio: float, csv_file: str, sample_rate: int = 16000):
        self.noise_ratio = noise_ratio
        self.sample_rate = sample_rate
        df = pd.read_csv(csv_file)
        self.rir_paths = df["path"].values

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def forward(self, audio: np.ndarray) -> np.ndarray:
        if np.random.random() > self.noise_ratio:
            return audio

        audio_power = float((audio**2).mean())
        if audio_power < 1e-10:
            return audio

        # Randomly select a RIR audio path and load it
        rir_path = np.random.choice(self.rir_paths)
        rir_audio, _ = librosa.load(rir_path, sr=self.sample_rate, mono=True)

        augmented = signal.convolve(audio, rir_audio, mode="full")[: audio.shape[0]]

        augment_power = float((augmented**2).mean())
        if augment_power > 1e-10:
            scale = float(np.sqrt(audio_power / augment_power))
            augmented = scale * augmented

        return augmented


@register_transform("rawboost")
class RawBoost:
    def __init__(self, noise_ratio, param2, param3, algo: int = 5, sample_rate: int = 16000):
        self.noise_ratio = noise_ratio
        self.param2 = param2  # Currently unused in the original function
        self.param3 = param3  # Currently unused in the original function
        self.algo = algo
        self.sample_rate = sample_rate
        self.parameters = get_default_args()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def forward(self, audio: np.ndarray) -> np.ndarray:
        # if np.random.random() > self.noise_ratio:
        #     return audio

        try:
            return process_Rawboost_feature(
                feature=audio,
                sr=self.sample_rate,
                args=self.parameters,
                algo=self.algo,
            )
        except Exception as e:
            print(f"Warning: RawBoost augmentation failed, returning original audio: {e}")
            return audio


@register_transform("codec")
class Codec:
    def __init__(self, noise_ratio: float, sample_rate: int = 16000):
        self.noise_ratio = noise_ratio
        self.sample_rate = sample_rate
        # formats = [("wav", "pcm_mulaw"), ("mp3", None), ("g722", None)]
        # TODO: add more formats
        self.formats = [("wav", "pcm_mulaw"), ("g722", None)]

    def __call__(self, x: dict) -> np.ndarray:
        return self.forward(x)

    def forward(self, x: dict) -> np.ndarray:
        if np.random.random() > self.noise_ratio:
            return x["audio"]

        audio: np.ndarray = x["audio"]
        fmt, enc = random.choice(self.formats)

        audio_tensor = torch.as_tensor(audio, dtype=torch.float32)
        audio_tensor = audio_tensor.unsqueeze(0).transpose(0, 1).cpu()

        eff = torchaudio.io.AudioEffector(format=fmt, encoder=enc)
        y = eff.apply(audio_tensor, self.sample_rate).transpose(0, 1).squeeze(0)

        out = y.numpy()
        if np.issubdtype(audio.dtype, np.floating):
            out = out.astype(audio.dtype, copy=False)
        return out


@register_transform("morph")
class Morph:
    def __init__(
        self, noise_ratio: float, noise_db_low: float = 5.0, noise_db_high: float = 20.0
    ) -> np.ndarray:
        self.noise_ratio = noise_ratio
        self.noise_db_low = noise_db_low
        self.noise_db_high = noise_db_high

    def __call__(self, x: dict) -> np.ndarray:
        return self.forward(x)

    def forward(self, x: dict) -> np.ndarray:
        if np.random.rand() > self.noise_ratio:
            return x["audio"]
        audio: np.ndarray = x["audio"]
        noise: np.ndarray = x["noise"]

        audio_power = float((audio**2).mean())
        noise_db = np.random.uniform(self.noise_db_low, self.noise_db_high)

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
class AddNoise:
    def __init__(self,
                 noise_ratio: float,
                 csv_file: str,
                 snr_low: int = 5,
                 snr_high: int = 20,
                 pad_noise: bool = False,
                 start_index: Optional[int] = None,
                 normalize: bool = False,
                 sample_rate: int = 16000):
        self.noise_ratio = noise_ratio
        self.snr_low = snr_low
        self.snr_high = snr_high
        self.pad_noise = pad_noise
        self.start_index = start_index
        self.normalize = normalize  # Currently unused in the original function
        self.sample_rate = sample_rate
        df = pd.read_csv(csv_file)
        self.noise_audio_paths = df["path"].values

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        return self.forward(audio)

    def forward(self, audio: np.ndarray) -> np.ndarray:
        # TODO: should we add a sample rate check?

        if np.random.random() > self.noise_ratio:
            return audio

        audio_power = float((audio**2).mean())
        if audio_power < 1e-10:
            logger.warning("Audio power is too low, likely a silent audio, returning original audio")
            return audio

        SNR = np.random.uniform(self.snr_low, self.snr_high)
        clean_amplitude = compute_amplitude(audio)
        noise_amplitude_factor = 1 / (dB_to_amplitude(SNR) + 1)
        new_noise_amplitude = noise_amplitude_factor * clean_amplitude
        noisy_waveform = audio * (1 - noise_amplitude_factor)

        # Randomly select a noise audio path and load it
        noise_path = np.random.choice(self.noise_audio_paths)
        noise, _ = librosa.load(noise_path, sr=self.sample_rate, mono=True)

        noise = align_waveform(noise, len(audio), self.pad_noise, self.start_index)

        noise_amplitude = compute_amplitude(noise)
        noise_waveform = noise * new_noise_amplitude / (noise_amplitude + 1e-14)
        noisy_waveform += noise_waveform

        return noisy_waveform


@register_transform("speed_perturb")
class SpeedPerturb:
    def __init__(self,
                 noise_ratio: float,
                 speeds: List[int] = [90, 100, 110],
                 perturb_prob: float = 1.0, # Currently unused in the original function
                 sample_rate: int = 16000):
        self.noise_ratio = noise_ratio
        self.speeds = speeds
        self.perturb_prob = perturb_prob
        self.sample_rate = sample_rate

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        return self.forward(audio)

    def forward(self, audio: np.ndarray) -> np.ndarray:
        if np.random.random() > self.noise_ratio:
            return audio

        speed = random.choice(self.speeds)
        new_freq = self.sample_rate * speed // 100
        resampled = librosa.resample(audio, orig_sr=self.sample_rate, target_sr=new_freq)

        return resampled


@register_transform("add_babble")
class AddBabble:
    def __init__(self,
                 noise_ratio: float,
                 csv_file: str,
                 speaker_count: int = 3,
                 snr_low: int = 0,
                 snr_high: int = 0,
                 pad_noise: bool = False,
                 start_index: Optional[int] = None,
                 sample_rate: int = 16000):
        self.noise_ratio = noise_ratio
        self.speaker_count = speaker_count
        self.snr_low = snr_low
        self.snr_high = snr_high
        self.pad_noise = pad_noise
        self.start_index = start_index
        self.sample_rate = sample_rate
        df = pd.read_csv(csv_file)
        self.babble_audio_paths = df["path"].values

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        return self.forward(audio)

    def forward(self, audio: np.ndarray) -> np.ndarray:
        if np.random.random() > self.noise_ratio:
            return audio

        # Randomly select speaker_count babble audio paths and load them
        replace = False
        if len(self.babble_audio_paths) < self.speaker_count:
            logger.warning(f"Less than {self.speaker_count} audio files in the csv file, using all of them")
            replace = True

        sampled_babble_paths = np.random.choice(self.babble_audio_paths, self.speaker_count, replace=replace)
        babble_waveforms = []
        for path in sampled_babble_paths:
            babble_audio, _ = librosa.load(path, sr=self.sample_rate, mono=True)
            babble_waveforms.append(babble_audio)
        
        SNR = np.random.uniform(self.snr_low, self.snr_high)
        clean_amplitude = compute_amplitude(audio)
        noise_amplitude_factor = 1 / (dB_to_amplitude(SNR) + 1)
        new_noise_amplitude = noise_amplitude_factor * clean_amplitude

        babbled_audio = audio * (1 - noise_amplitude_factor)

        audio_len = len(audio)
        babble_waveform = np.zeros(audio_len, dtype=audio.dtype)
        for i in range(self.speaker_count):
            waveform_idx = (1 + i) % self.speaker_count
            aligned_bw = align_waveform(
                babble_waveforms[waveform_idx],
                audio_len, self.pad_noise, self.start_index)
            babble_waveform += aligned_bw

        babble_amplitude = compute_amplitude(babble_waveform)
        babble_waveform *= new_noise_amplitude / (babble_amplitude + 1e-14)
        babbled_audio += babble_waveform

        return babbled_audio

@register_transform("drop_freq")
class DropFreq:
    def __init__(self,
                 noise_ratio: float,
                 drop_freq_low: float = 1e-14,
                 drop_freq_high: float = 1,
                 drop_count_low: int = 1,
                 drop_count_high: int = 2,
                 drop_width: float = 0.05,
                 sample_rate: int = 16000) -> np.ndarray:
        self.noise_ratio = noise_ratio
        self.drop_freq_low = drop_freq_low
        self.drop_freq_high = drop_freq_high
        self.drop_count_low = drop_count_low
        self.drop_count_high = drop_count_high
        self.drop_width = drop_width
        self.sample_rate = sample_rate

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        return self.forward(audio)

    def forward(self, audio: np.ndarray) -> np.ndarray:
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
        if np.random.random() > self.noise_ratio:
            return audio

        drop_count = np.random.randint(self.drop_count_low, self.drop_count_high + 1)
        drop_range = self.drop_freq_high - self.drop_freq_low
        drop_frequencies = np.random.rand(drop_count) * drop_range + self.drop_freq_low

        # Filter parameters, hard coded just like speechbrain's impl
        filter_length = 101
        pad = filter_length // 2

        # create a delta filter
        drop_filter = np.zeros(filter_length)
        drop_filter[pad] = 1 # impulse

        nyquist = self.sample_rate / 2
        for frequency in drop_frequencies:
            notch_kernel = notch_filter(frequency, filter_length, self.drop_width)
            drop_filter = np.convolve(drop_filter, notch_kernel, mode='same')

        dropped_audio = np.convolve(audio, drop_filter, mode='same')

        return dropped_audio


@register_transform("drop_chunk")
class DropChunk:
    def __init__(self,
                 noise_ratio: float,
                 drop_length_low: int = 100,
                 drop_length_high: int = 1000,
                 drop_count_low: int = 1,
                 drop_count_high: int = 10,
                 drop_start: int = 0,
                 drop_end: Optional[int] = None,
                 noise_factor: float = 0.0) -> np.ndarray:
        self.noise_ratio = noise_ratio
        self.drop_length_low = drop_length_low
        self.drop_length_high = drop_length_high
        self.drop_count_low = drop_count_low
        self.drop_count_high = drop_count_high
        self.drop_start = drop_start
        self.drop_end = drop_end
        self.noise_factor = noise_factor

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        return self.forward(audio)

    def forward(self, audio: np.ndarray) -> np.ndarray:
        if np.random.random() > self.noise_ratio:
            return audio

        audio_len = len(audio)
        dropped_audio = audio.copy()

        clean_amplitude = compute_amplitude(audio)

        drop_times = np.random.randint(self.drop_count_low, self.drop_count_high + 1)

        for _ in range(drop_times):
            drop_length = np.random.randint(self.drop_length_low, self.drop_length_high + 1)

            start_min = self.drop_start
            if start_min < 0:
                start_min += audio_len

            start_max = self.drop_end if self.drop_end is not None else audio_len
            if start_max < 0:
                start_max += audio_len
            start_max = max(0, start_max - drop_length)

            if start_min >= start_max:
                continue

            start = np.random.randint(start_min, start_max + 1)
            end = min(start + drop_length, audio_len)

            if self.noise_factor == 0.0:
                dropped_audio[start:end] = 0.0
            else:
                noise_max = 2 * clean_amplitude * self.noise_factor
                noise_vec = np.random.rand(end - start) * 2 * noise_max - noise_max
                dropped_audio[start:end] = noise_vec

        return dropped_audio



@register_transform("do_clip")
class DoClip:
    def __init__(self,
                 noise_ratio: float,
                 clip_low: float = 0.5,
                 clip_high: float = 1,
                 clip_prob: float = 1):
        self.noise_ratio = noise_ratio
        self.clip_low = clip_low
        self.clip_high = clip_high
        self.clip_prob = clip_prob # Currently unused in the original function

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        return self.forward(audio)

    def forward(self, audio: np.ndarray) -> np.ndarray:
        if np.random.random() > self.noise_ratio:
            return audio

        clipping_range = self.clip_high - self.clip_low
        clip_value = np.random.rand() * clipping_range + self.clip_low
        clipped_audio = np.clip(audio, -clip_value, clip_value)

        return clipped_audio
