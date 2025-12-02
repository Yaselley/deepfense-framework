# Augmentations

Located in `deepfense.data.transforms.augmentations`. These classes are registered via `@register_transform`.

## AugmentationPipeline

`deepfense.data.transforms.augmentations.AugmentationPipeline`

Orchestrates the application of multiple transforms.

**Args**:
*   `transforms` (List[Dict]): List of config dicts for child transforms.
*   `mode` (str):
    *   `sequential`: Apply all (or `k`) transforms in order.
    *   `parallel`: Randomly select **one** transform from the list (OneOf).
*   `k` (int): Number of transforms to apply in sequential mode.
*   `p` (float): Global probability of applying the pipeline.
*   `execution` (str):
    *   `chain`: `x = T2(T1(x))` (Default).
    *   `independent`: Returns list `[T1(x), T2(x)]` (used with `concat_original`).
*   `concat_original` (bool): If True, returns `[Original, Augmented]`.

## Transform Classes

All transforms accept a `noise_ratio` (probability of application).

### RawBoost
`deepfense.data.transforms.augmentations.RawBoost`

Applies linear and non-linear convolutive noise to simulate transmission channel distortion.
*   **Args**: `algo` (int) - Algorithm ID (1-8) per the RawBoost paper.

### RIR (Reverb)
`deepfense.data.transforms.augmentations.RIR`

Convolves audio with a Room Impulse Response (RIR).
*   **Args**: `csv_file` (str) - Path to CSV containing paths to RIR audio files.

### AddNoise
`deepfense.data.transforms.augmentations.AddNoise`

Adds additive noise (MUSAN, etc.) at a random SNR.
*   **Args**:
    *   `csv_file` (str): Path to noise files CSV.
    *   `snr_low` (float): Min SNR.
    *   `snr_high` (float): Max SNR.

### Codec
`deepfense.data.transforms.augmentations.Codec`

Simulates audio codec compression (e.g., mp3, ogg, mulaw).
*   **Args**: `sample_rate` (int).

### SpeedPerturb
`deepfense.data.transforms.augmentations.SpeedPerturb`

Resamples audio to change speed/pitch.
*   **Args**: `speeds` (List[int]) - Percentages (e.g., `[90, 100, 110]`).

### AddBabble
`deepfense.data.transforms.augmentations.AddBabble`

Mixes multiple speakers ("babble noise") into the audio.
*   **Args**: `speaker_count` (int).

