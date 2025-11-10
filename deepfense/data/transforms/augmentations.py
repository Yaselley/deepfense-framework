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

if __name__ == "__main__":
    print("Waveform augmentation tools loaded")
