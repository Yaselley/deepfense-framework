import os
import sys
import copy
import math
import numpy as np
import logging
import pandas as pd
import soundfile as sf
import torchaudio
import librosa
import random
from typing import Optional, List
import torch
import torch.nn.functional as F
from scipy import signal
from deepfense.utils.registry import register_transform, build_transform
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
class SimpleAug:
    def __init__(self, noise_ratio: float = 0.0):
        self.noise_ratio = noise_ratio

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x


@register_transform("rir")
class RIR:
    def __init__(self, noise_ratio: float, csv_file: str, sample_rate: int = 16000):
        self.noise_ratio = noise_ratio
        self.csv_file = csv_file
        self.sample_rate = sample_rate
        
        # Pre-load CSV for efficiency
        try:
            self.df = pd.read_csv(self.csv_file)
            if "path" not in self.df.columns:
                 self.df = pd.read_csv(self.csv_file, header=None, names=["path"])
        except Exception as e:
            logger.error(f"Failed to load RIR CSV {self.csv_file}: {e}")
            self.df = [] # Empty list fallback

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.random.random() > self.noise_ratio:
            return x
        
        if len(self.df) == 0:
             return x

        audio = x
        # Pass pre-loaded DataFrame
        rir_audio = select_audio(self.df, self.sample_rate)

        audio_power = float((audio**2).mean())
        if audio_power < 1e-10:
            return audio

        augmented = signal.convolve(audio, rir_audio, mode="full")[: audio.shape[0]]

        augment_power = float((augmented**2).mean())
        if augment_power > 1e-10:
            scale = float(np.sqrt(audio_power / augment_power))
            augmented = scale * augmented
        
        return augmented


@register_transform("rawboost")
class RawBoost:
    def __init__(self, noise_ratio: float, algo: int = 5, **kwargs):
        self.noise_ratio = noise_ratio
        self.algo = algo
        self.sample_rate = 16000 # Fixed in original code
        self.params = get_default_args()
        # Allow overriding default args from config if needed
        for k, v in kwargs.items():
            setattr(self.params, k, v)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.random.random() > self.noise_ratio:
            return x
        
        audio = x
        try:
            return process_Rawboost_feature(
                feature=audio,
                sr=self.sample_rate,
                args=self.params,
                algo=self.algo,
            )
        except Exception as e:
            return audio


@register_transform("codec")
class Codec:
    def __init__(self, noise_ratio: float, sample_rate: int = 16000):
        self.noise_ratio = noise_ratio
        self.sample_rate = sample_rate
        self.formats = [("wav", "pcm_mulaw"), ("g722", None)]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.random.random() > self.noise_ratio:
            return x

        audio = x
        fmt, enc = random.choice(self.formats)

        x_t = torch.as_tensor(audio, dtype=torch.float32)
        x_t = x_t.unsqueeze(0).transpose(0, 1).cpu()

        try:
            eff = torchaudio.io.AudioEffector(format=fmt, encoder=enc)
            y = eff.apply(x_t, self.sample_rate).transpose(0, 1).squeeze(0)

            out = y.numpy()
            if np.issubdtype(audio.dtype, np.floating):
                out = out.astype(audio.dtype, copy=False)
            return out
        except Exception as e:
            # Fallback if codec fails (e.g. system missing libs)
            return audio


@register_transform("morph")
class Morph:
    def __init__(self, noise_ratio: float, **kwargs):
        self.noise_ratio = noise_ratio
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x


@register_transform("add_noise")
class AddNoise:
    def __init__(self, 
                 noise_ratio: float, 
                 csv_file: str, 
                 snr_low: float = 5, 
                 snr_high: float = 20, 
                 pad_noise: bool = False, 
                 normalize: bool = False, 
                 sample_rate: int = 16000):
        self.noise_ratio = noise_ratio
        self.csv_file = csv_file
        self.snr_low = snr_low
        self.snr_high = snr_high
        self.pad_noise = pad_noise
        self.normalize = normalize
        self.sample_rate = sample_rate
        
        # Pre-load CSV
        try:
            self.df = pd.read_csv(self.csv_file)
            if "path" not in self.df.columns:
                 self.df = pd.read_csv(self.csv_file, header=None, names=["path"])
        except Exception as e:
            logger.error(f"Failed to load Noise CSV {self.csv_file}: {e}")
            self.df = []

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.random.random() > self.noise_ratio:
            return x
        
        if len(self.df) == 0:
            return x

        audio = x
        audio_power = float((audio**2).mean())
        if audio_power < 1e-10:
            return audio
        
        SNR = np.random.uniform(self.snr_low, self.snr_high)
        clean_amplitude = compute_amplitude(audio) 
        noise_amplitude_factor = 1 / (dB_to_amplitude(SNR) + 1)
        new_noise_amplitude = noise_amplitude_factor * clean_amplitude
        noisy_waveform = audio * (1 - noise_amplitude_factor)

        noise = select_audio(self.df, self.sample_rate)
        audio_len = len(audio)
        # start_index is None -> random
        noise = align_waveform(noise, audio_len, self.pad_noise, start_index=None)
        
        noise_amplitude = compute_amplitude(noise)
        noise_waveform = noise * new_noise_amplitude / (noise_amplitude + 1e-14)
        noisy_waveform += noise_waveform

        return noisy_waveform


@register_transform("speed_perturb")
class SpeedPerturb:
    def __init__(self, noise_ratio: float, speeds: List[int] = None, sample_rate: int = 16000):
        self.noise_ratio = noise_ratio
        self.speeds = speeds if speeds else [90, 100, 110]
        self.sample_rate = sample_rate

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.random.random() > self.noise_ratio:
            return x
        
        speed = random.choice(self.speeds)
        new_freq = self.sample_rate * speed // 100
        resampled = librosa.resample(x, orig_sr=self.sample_rate, target_sr=new_freq)
        return resampled


@register_transform("add_babble")
class AddBabble:
    def __init__(self, 
                 noise_ratio: float, 
                 csv_file: str, 
                 speaker_count: int = 3, 
                 snr_low: float = 0, 
                 snr_high: float = 0, 
                 pad_noise: bool = False,
                 sample_rate: int = 16000):
        self.noise_ratio = noise_ratio
        self.csv_file = csv_file
        self.speaker_count = speaker_count
        self.snr_low = snr_low
        self.snr_high = snr_high
        self.pad_noise = pad_noise
        self.sample_rate = sample_rate
        
        # Pre-load CSV
        try:
            self.df = pd.read_csv(self.csv_file)
            if "path" not in self.df.columns:
                 self.df = pd.read_csv(self.csv_file, header=None, names=["path"])
        except Exception as e:
            logger.error(f"Failed to load Babble CSV {self.csv_file}: {e}")
            self.df = []

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.random.random() > self.noise_ratio:
            return x
        
        if len(self.df) == 0:
            return x

        audio = x
        babble_waveforms = select_multiple_audio(
            self.df, 
            self.speaker_count, 
            self.sample_rate)
        
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
                audio_len, self.pad_noise, start_index=None)
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
                 sample_rate: int = 16000):
        self.noise_ratio = noise_ratio
        self.drop_freq_low = drop_freq_low
        self.drop_freq_high = drop_freq_high
        self.drop_count_low = drop_count_low
        self.drop_count_high = drop_count_high
        self.drop_width = drop_width
        self.sample_rate = sample_rate

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.random.random() > self.noise_ratio:
            return x
        
        audio = x
        drop_count = np.random.randint(self.drop_count_low, self.drop_count_high + 1)
        drop_range = self.drop_freq_high - self.drop_freq_low
        drop_frequencies = np.random.rand(drop_count) * drop_range + self.drop_freq_low

        # Filter parameters
        filter_length = 101
        pad = filter_length // 2

        # create a delta filter 
        drop_filter = np.zeros(filter_length)
        drop_filter[pad] = 1 # impulse
        
        # nyquist = self.sample_rate / 2
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
                 noise_factor: float = 0.0):
        self.noise_ratio = noise_ratio
        self.drop_length_low = drop_length_low
        self.drop_length_high = drop_length_high
        self.drop_count_low = drop_count_low
        self.drop_count_high = drop_count_high
        self.drop_start = drop_start
        self.drop_end = drop_end
        self.noise_factor = noise_factor

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.random.random() > self.noise_ratio:
            return x
        
        audio = x
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

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.random.random() > self.noise_ratio:
            return x
        
        clipping_range = self.clip_high - self.clip_low
        clip_value = np.random.rand() * clipping_range + self.clip_low
        clipped_audio = np.clip(x, -clip_value, clip_value)
        
        return clipped_audio


@register_transform("augmentation_pipeline")
class AugmentationPipeline:
    def __init__(self, 
                 transforms: list, 
                 mode: str = "sequential", 
                 k: Optional[int] = None, 
                 p: float = 1.0,
                 concat_original: bool = False,
                 execution: str = "chain"):
        """
        Args:
            transforms (list): List of transform configs (dicts).
            mode (str): "sequential" or "parallel" (determines SELECTION strategy).
                        "sequential": Selects 'k' (or all) transforms.
                        "parallel": Selects 1 transform (OneOf).
            k (int, optional): Number of transforms to select if mode="sequential".
            p (float): Probability of applying the pipeline.
            concat_original (bool): If True, returns [Original, Result].
            execution (str): "chain" or "independent".
                             "chain": Applies selected transforms in sequence to the same audio.
                             "independent": Applies selected transforms separately to copies of original audio.
        """
        self.transforms_configs = transforms
        self.mode = mode.lower() # Selection Strategy
        self.k = k
        self.p = p
        self.concat_original = concat_original
        self.execution = execution.lower() # Application Strategy
        
        self.loaded_transforms = []
        for cfg in self.transforms_configs:
            cfg_copy = cfg.copy()
            t_type = cfg_copy.pop("type")
            self.loaded_transforms.append(build_transform(t_type, **cfg_copy))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.random.random() > self.p:
            return x
        
        if not self.loaded_transforms:
            return x

        # 1. Selection Phase
        if self.mode == "parallel":
            # Parallel Selection: Pick exactly 1
            selected = [random.choice(self.loaded_transforms)]
        elif self.mode == "sequential":
            # Sequential Selection: Pick K (or all)
            if self.k is None:
                selected = self.loaded_transforms
            else:
                k_select = min(self.k, len(self.loaded_transforms))
                selected = random.sample(self.loaded_transforms, k_select)
        else:
             # Default fallback
             selected = self.loaded_transforms

        # 2. Execution Phase
        target_len = x.shape[0]
        
        if self.execution == "independent":
            # Apply each selected transform separately to a copy of x
            # Returns list of results
            augmented_results = []
            for t in selected:
                res = t(x.copy())
                # Ensure length consistency for stacking
                if len(res) != target_len:
                    res = align_waveform(res, target_len, pad_noise=False)
                augmented_results.append(res)
            
            # If concat_original is True, prepend original
            if self.concat_original:
                final_list = [x] + augmented_results
            else:
                final_list = augmented_results
                
            # Note: This returns a numpy stack, effectively increasing batch size dim 
            # if collate_fn handles it (it does in deepfense/data/data_utils.py)
            return np.stack(final_list)

        else: # execution == "chain"
            # Apply selected transforms in sequence to x
            out = x.copy()
            for t in selected:
                out = t(out)
            
            if self.concat_original:
                if len(out) != target_len:
                    out = align_waveform(out, target_len, pad_noise=False)
                return np.stack([x, out])
            
            return out
