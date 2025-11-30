# Data Pipeline & Augmentations

DeepFense provides a robust data pipeline capable of loading huge datasets via Parquet and applying complex augmentation chains.

## Data Processing Flow
1.  **Load Audio**: Audio is loaded from disk (WAV/FLAC/MP3).
2.  **Resample**: If needed, audio is resampled to `data.sampling_rate`.
3.  **Base Transform**: Basic operations like Padding/Trimming (always applied).
4.  **Augment Transform**: Complex, probabilistic augmentations (Training only).
5.  **Collate**: Batching and Mask creation.

---

## Base Transforms

These are deterministic operations usually applied to both Train and Val sets.

### 1. Load Audio (`load_audio`)
*   **Arguments**:
    *   `target_sr`: Target sampling rate.
    *   `mono`: If true, converts stereo to mono (mean).

### 2. Pad/Truncate (`pad`)
Ensures all audio clips are exactly `max_len` samples long.
*   **Arguments**:
    *   `max_len` (int): Target length in samples.
    *   `random_pad` (bool): If `True` (and audio > max_len), picks a random crop. If `False`, takes the start.
    *   `pad_type` (str): Strategy for short audio. Currently supports `"repeat"` (tiles the audio).

---

## Augmentation Pipeline

Defined in `augment_transform`.

### Structure & Configuration

The `augmentation_pipeline` is a flexible container that controls *how* multiple augmentations are selected and applied.

*   **`mode`** (Selection Strategy):
    *   `"sequential"`: Selects **ALL** transforms in the list (or `k` items if specified).
    *   `"parallel"`: Selects exactly **ONE** transform from the list randomly (OneOf).
*   **`execution`** (Application Strategy):
    *   `"chain"`: Applies selected transforms **in sequence** to the *same* audio object (A -> B -> C).
    *   `"independent"`: Applies each selected transform to a **fresh copy** of the original audio (Branching).
*   **`concat_original`**:
    *   If `True`, the original clean audio is preserved and prepended to the results.
    *   **Note**: This effectively increases the batch size during training.

### Common Configurations

| Goal | Mode | Execution | Concat Orig | Output Size | Result |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard Augmentation** | `parallel` | `chain` | `False` | 1 | **[Augmented]** (Either A or B) |
| **Data Expansion (1 extra)** | `parallel` | `chain` | `True` | 2 | **[Original, Augmented]** |
| **Data Expansion (All variations)** | `sequential` | `independent` | `True` | N+1 | **[Original, Aug_A, Aug_B]** |
| **Sequential Chain** | `sequential` | `chain` | `False` | 1 | **[Augmented]** (A applied, then B) |

#### Example 1: Randomly apply ONE augmentation (RawBoost OR RIR) keeping the original
```yaml
type: augmentation_pipeline
mode: parallel           # Pick 1
concat_original: true    # Keep Original
transforms:
  - {type: rawboost, ...}
  - {type: rir, ...}
# Output: [Original, RawBoost_ver] OR [Original, RIR_ver]
```

#### Example 2: Generate separate versions for ALL augmentations (RawBoost AND RIR)
```yaml
type: augmentation_pipeline
mode: sequential         # Pick All
execution: independent   # Branching
concat_original: true    # Keep Original
transforms:
  - {type: rawboost, ...}
  - {type: rir, ...}
# Output: [Original, RawBoost_ver, RIR_ver]
```

*   **`p`**: Probability of running the entire pipeline.

### Available Augmentations

#### 1. RawBoost (`rawboost`)
Adds linear and non-linear convolutive noise and impulsive noise.
*   **noise_ratio** (float): Probability (0-1).
*   **algo** (int): Algorithm ID (0-5).

#### 2. RIR / Reverb (`rir`)
Convolves audio with Room Impulse Responses.
*   **noise_ratio** (float): Probability.
*   **csv_file** (str): Path to CSV containing paths to RIR wav files.

#### 3. Add Noise (`add_noise`)
Adds additive background noise.
*   **noise_ratio** (float): Probability.
*   **csv_file** (str): Path to noise CSV.
*   **snr_low** / **snr_high** (float): Range of Signal-to-Noise Ratio to apply (in dB).
*   **pad_noise** (bool): If True, tiles noise to match audio length.

#### 4. Add Babble (`add_babble`)
Mixes multiple speakers ("babble") into the background.
*   **noise_ratio** (float): Probability.
*   **csv_file** (str): Path to speech CSV.
*   **speaker_count** (int): Number of speakers to mix (default: 3).
*   **snr_low** / **snr_high**: Mixing SNR range.

#### 5. Speed Perturbation (`speed_perturb`)
Resamples audio to change pitch and speed.
*   **noise_ratio** (float): Probability.
*   **speeds** (list[int]): List of percentages, e.g., `[90, 100, 110]` (90% speed, 100% speed, etc.).

#### 6. Codec Compression (`codec`)
Simulates compression artifacts.
*   **noise_ratio** (float): Probability.
*   **formats**: Hardcoded to random choice of `("wav", "pcm_mulaw")` or `("g722", None)`.

#### 7. Drop Frequencies (`drop_freq`)
Applies random notch filters to drop frequency bands.
*   **noise_ratio** (float): Probability.
*   **drop_freq_low/high** (float): Normalized frequency range (0-1).
*   **drop_count_low/high** (int): Number of notches to apply.

#### 8. Drop Chunk (`drop_chunk`)
Zeros out (or replaces with noise) random time segments.
*   **noise_ratio** (float): Probability.
*   **drop_length_low/high** (int): Length of chunks in samples.
*   **drop_count_low/high** (int): Number of chunks.
*   **noise_factor** (float): If > 0, fills chunk with random noise scaled by this factor.

#### 9. Clipping (`do_clip`)
Clips signal amplitude to simulate saturation.
*   **noise_ratio** (float): Probability.
*   **clip_low/high** (float): Range of clipping thresholds.
