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


# Add more as needed (codec, reverb, morph, RawBoost etc.)
# Just wrap each function with @register_transform("name")

if __name__ == "__main__":
    print("Waveform augmentation tools loaded")
