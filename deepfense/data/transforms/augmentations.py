import os
import sys
import copy
import numpy as np
from scipy import signal
from pathlib import Path
from pydub import AudioSegment
from deepfense.data.transforms.registry import register_transform


@register_transform("simple_aug")
def sample_aug_func(x, noise_ratio):
    return x


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
