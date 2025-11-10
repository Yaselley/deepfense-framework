import os
import sys
import copy
import numpy as np
from scipy import signal
from pathlib import Path
from pydub import AudioSegment
from deepfense.data.transforms.registry import register_transform


@register_transform("wav_time_mask")
def wav_time_mask(input_data, wav_samp_rate):
    """Randomly zero-out a segment in time domain."""
    seg_width = int(np.random.rand() * 0.2 * wav_samp_rate)
    start_idx = int(np.random.rand() * max(1, input_data.shape[0] - seg_width))
    tmp = np.ones_like(input_data)
    tmp[start_idx:start_idx + seg_width] = 0
    return input_data * tmp


@register_transform("batch_siltrim_for_multiview")
def batch_siltrim_for_multiview(input_data_batch, wav_samp_rate,
                                random_trim_sil=False, random_trim_nosil=False):
    _, start_time, end_time = wav_rand_sil_trim(
        input_data_batch[0], wav_samp_rate, random_trim_sil, random_trim_nosil
    )
    if start_time < end_time and start_time > 0:
        return [data[start_time:end_time] for data in input_data_batch]
    return input_data_batch


@register_transform("batch_pad_for_multiview")
def batch_pad_for_multiview(input_data_batch_, wav_samp_rate, length,
                            random_trim_nosil=False, repeat_pad=False):
    def _ad_length(x, length, repeat_pad):
        if length > x.shape[0]:
            if repeat_pad:
                rt = int(np.ceil(length / x.shape[0]))
                return np.tile(x, (rt, 1))[:length]
            tmp = np.zeros([length, 1])
            tmp[:x.shape[0]] = x
            return tmp
        return x[:length]

    firstlen = input_data_batch_[0].shape[0]
    input_data_batch = [_ad_length(x, firstlen, repeat_pad) for x in input_data_batch_]
    new_len = input_data_batch[0].shape[0]

    if repeat_pad is False:
        start_len = int(np.random.rand() * (new_len - length)) if random_trim_nosil else 0
        end_len = start_len + length
        input_data_batch_ = input_data_batch
    else:
        if new_len < length:
            rt = int(np.ceil(length / new_len))
            input_data_batch_ = [np.tile(x, (rt, 1)) for x in input_data_batch]
            start_len, end_len = 0, length
        else:
            start_len = int(np.random.rand() * (new_len - length)) if random_trim_nosil else 0
            end_len = start_len + length
            input_data_batch_ = input_data_batch

    return [x[start_len:end_len] for x in input_data_batch_]


################
# Frequency domain
################
@register_transform("wav_freq_pass_fixed")
def wav_freq_pass_fixed(input_data, wav_samp_rate, start_b, end_b):
    filter_order = 10
    if start_b < 0.01:
        sos = signal.butter(filter_order, end_b, 'lowpass', output='sos')
    elif end_b > 0.99:
        sos = signal.butter(filter_order, start_b, 'highpass', output='sos')
    else:
        sos = signal.butter(filter_order, [start_b, end_b], 'bandpass', output='sos')

    filtered = signal.sosfilt(sos, input_data[:, 0])
    return np.expand_dims(filtered, axis=1)


@register_transform("wav_freq_mask_fixed")
def wav_freq_mask_fixed(input_data, wav_samp_rate, start_b, end_b):
    filter_order = 10
    if start_b < 0.01:
        sos = signal.butter(filter_order, end_b, 'highpass', output='sos')
    elif end_b > 0.99:
        sos = signal.butter(filter_order, start_b, 'lowpass', output='sos')
    else:
        sos = signal.butter(filter_order, [start_b, end_b], 'bandstop', output='sos')

    filtered = signal.sosfilt(sos, input_data[:, 0])
    return np.expand_dims(filtered, axis=1)


@register_transform("wav_freq_mask")
def wav_freq_mask(input_data, wav_samp_rate):
    max_band_witdh = 0.2
    band_w = np.random.rand() * max_band_witdh
    if band_w < 0.05:
        return input_data
    start_b = np.random.rand() * (1 - band_w)
    end_b = start_b + band_w
    return wav_freq_mask_fixed(input_data, wav_samp_rate, start_b, end_b)


################
# Padding
################
@register_transform("pad")
def pad_combined(x, max_len=64600, random_pad=False, pad_type="repeat"):
    x_len = x.shape[0]
    if x_len > max_len:
        if random_pad:
            start = np.random.randint(0, x_len - max_len)
            return x[start:start + max_len]
        return x[:max_len]

    pad_len = max_len - x_len
    if pad_type == "repeat":
        repeats = int(np.ceil(max_len / x_len))
        return np.tile(x, repeats)[:max_len]
    elif pad_type == "zero":
        padded = np.zeros(max_len, dtype=x.dtype)
        padded[:x_len] = x
        return padded
    raise ValueError(f"Unknown pad_type: {pad_type}")

# TODO: We might need try except to catch the exceptions in case
# augmentation failed (e.g., failed to load the rir due to some network issue)
@register_transform("rir")
def rir(
    audio: np.ndarray,
    rir_audio: np.ndarray,
    #rir_path: str, # TODO: rir_audio or a path?
    probability: float = 0.3,
) -> np.ndarray:
    """
    apply rir augmentation to monochannel audio 
    - rir_path: directory containing .wav RIR files
    """
    prob = np.random.rand()
    if prob > probability:
        return audio

    audio_power = float((audio ** 2).mean())
    if audio_power < 1e-10:
        return audio

    #rir, sample_rate = audio_util.get_audio(rir_path)  # assume mono, no trim

    augmented = signal.convolve(audio, rir_audio, mode="full")[: audio.shape[0]]

    augment_power = float((augmented ** 2).mean())
    if augment_power > 1e-10:
        scale = float(np.sqrt(audio_power / augment_power))
        augmented = scale * augmented

    return augmented


from deepfense.data.transforms.RawBoost.data_utils_rawboost import process_Rawboost_feature, get_default_args
@register_transform("rawboost")
def rawboost(
    audio: np.ndarray,
    algorithm: int = 5,
    probability: float = 0.5,
    sample_rate: int = 16000,
) -> np.ndarray:
    """
    apply RawBoost augmentation to mono audio 
    """
    if np.random.random() > probability:
        return audio

    if parameters is None:
        parameters = get_default_args()

    try:
        return process_Rawboost_feature(
            feature=audio,
            sr=sample_rate,
            args=parameters,
            algo=algorithm,
        )
    except Exception as e:
        print(f"Warning: RawBoost augmentation failed, returning original audio: {e}")
        return audio


import torchaudio
import random
import torch
@register_transform("codec")
def codec(audio: np.ndarray, 
    sample_rate: int = 16000) -> np.ndarray:
    """
    Apply codec augmentation on mono audio. 
    Implementation adapted from speechbrain.
    """
    #formats = [("wav", "pcm_mulaw"), ("mp3", None), ("g722", None)]
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
    audio: np.ndarray,
    noise: np.ndarray,
    noise_db_low: float = 5,
    noise_db_high: float = 20,
) -> np.ndarray:
    """
    Apply morphing augmentation on mono audio.
    Implementation adapted from ESPNet.
    """
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
        noise = noise[offset:offset + audio_nsamples]

    noise_power = float((noise ** 2).mean())
    scale = (
            10 ** (-noise_db / 20) 
            * np.sqrt(audio_power)
            / np.sqrt(np.maximum(noise_power, 1e-10))
        )
    audio = audio + scale * noise
    return audio

# Add more as needed (codec, reverb, morph, RawBoost etc.)
# Just wrap each function with @register_transform("name")
# TODO: Speechbrain supports many? Add speechbrain lib?

if __name__ == "__main__":
    print("Waveform augmentation tools loaded")
    
    import os, sys, numpy as np, soundfile as sf
    from pathlib import Path
    import librosa

    audio_path = Path(sys.argv[1])
    outdir = Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)

    x, sr = librosa.load(audio_path, sr=16000, mono=True) 

    rir_len = min(len(x), 4096)
    t = np.arange(rir_len)
    rir_audio = (np.random.randn(rir_len) * np.exp(-t / (0.1 * sr))).astype(np.float32)

    noise = np.random.randn(len(x))

    y_rir = rir(x, rir_audio=rir_audio, probability=1.0)
    sf.write(outdir / f"{audio_path.stem}_rir.wav", y_rir, sr)

    y_raw = rawboost(x, probability=0.0, sample_rate=sr)
    sf.write(outdir / f"{audio_path.stem}_rawboost.wav", y_raw, sr)

    y_codec = codec(x, sample_rate=sr)
    sf.write(outdir / f"{audio_path.stem}_codec.wav", y_codec, sr)

    y_morph = morph(x, noise=noise, noise_db_low=5, noise_db_high=20)
    sf.write(outdir / f"{audio_path.stem}_morph.wav", y_morph, sr)
