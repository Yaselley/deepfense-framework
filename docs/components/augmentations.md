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

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `target_sr` | int | Target sampling rate. |
| `mono` | bool | If true, converts stereo to mono (mean). |

### 2. Pad/Truncate (`pad`)
Ensures all audio clips are exactly `max_len` samples long.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `max_len` | int | Target length in samples. |
| `random_pad` | bool | If `True` (and audio > max_len), picks a random crop. If `False`, takes the start. |
| `pad_type` | str | Strategy for short audio. Currently supports `"repeat"` (tiles the audio). |

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

### 1. RawBoost (`rawboost`)
Adds linear and non-linear convolutive noise and impulsive noise.

**Configuration Signature:**
```yaml
- type: rawboost
  args:
    noise_ratio: float
    algo: int
```

**Parameters:**

* **noise_ratio** - (*float*) Probability of applying this augmentation (0-1).
* **algo** - (*int*) Algorithm ID (0-5). Controls the type of boost.

**Example:**
```yaml
- type: rawboost
  noise_ratio: 0.5
  algo: 5
```

---

### 2. RIR / Reverb (`rir`)
Convolves audio with Room Impulse Responses.

**Configuration Signature:**
```yaml
- type: rir
  args:
    noise_ratio: float
    csv_file: string (csv file of paths, df["path"] = [path1, path2, ...])
```

**Parameters:**

* **noise_ratio** - (*float*) Probability of applying this augmentation.
* **csv_file** - (*str*) Path to CSV containing paths to RIR wav files.
  * **Format**: CSV with a 'path' column containing absolute paths to audio files.

**Example:**
```yaml
- type: rir
  noise_ratio: 0.8
  csv_file: "/path/to/rirs.csv"
```

---

### 3. Add Noise (`add_noise`)
Adds additive background noise.

**Configuration Signature:**
```yaml
- type: add_noise
  args:
    noise_ratio: float
    csv_file: string
    snr_low: float
    snr_high: float
    pad_noise: bool
```

**Parameters:**

* **noise_ratio** - (*float*) Probability of applying this augmentation.
* **csv_file** - (*str*) Path to CSV containing paths to noise audio files.
  * **Format**: CSV with a 'path' column containing absolute paths to audio files.
* **snr_low** - (*float*) Minimum Signal-to-Noise Ratio (dB).
* **snr_high** - (*float*) Maximum Signal-to-Noise Ratio (dB).
* **pad_noise** - (*bool*) If `True`, tiles noise to match audio length.

**Example:**
```yaml
- type: add_noise
  noise_ratio: 0.5
  csv_file: "/path/to/noises.csv"
  snr_low: 0
  snr_high: 20
```

---

### 4. Add Babble (`add_babble`)
Mixes multiple speakers ("babble") into the background.

**Configuration Signature:**
```yaml
- type: add_babble
  args:
    noise_ratio: float
    csv_file: string
    speaker_count: int
    snr_low: float
    snr_high: float
```

**Parameters:**

* **noise_ratio** - (*float*) Probability of applying this augmentation.
* **csv_file** - (*str*) Path to CSV containing paths to speech audio files.
  * **Format**: CSV with a 'path' column containing absolute paths to audio files.
* **speaker_count** - (*int*) Number of speakers to mix (default: 3).
* **snr_low** - (*float*) Minimum Mixing SNR (dB).
* **snr_high** - (*float*) Maximum Mixing SNR (dB).

**Example:**
```yaml
- type: add_babble
  noise_ratio: 0.3
  csv_file: "/path/to/speech.csv"
  speaker_count: 5
  snr_low: 10
  snr_high: 30
```

---

### 5. Speed Perturbation (`speed_perturb`)
Resamples audio to change pitch and speed.

**Configuration Signature:**
```yaml
- type: speed_perturb
  args:
    noise_ratio: float
    speeds: list[int]
```

**Parameters:**

* **noise_ratio** - (*float*) Probability of applying this augmentation.
* **speeds** - (*list[int]*) List of percentages, e.g., `[90, 100, 110]` (90% speed, 100% speed, etc.).

**Example:**
```yaml
- type: speed_perturb
  noise_ratio: 0.5
  speeds: [90, 95, 100, 105, 110]
```

---

### 6. Codec Compression (`codec`)
Simulates compression artifacts.

**Configuration Signature:**
```yaml
- type: codec
  args:
    noise_ratio: float
    formats: list
```

**Parameters:**

* **noise_ratio** - (*float*) Probability of applying this augmentation.
* **formats** - (*list*) Hardcoded to random choice of `("wav", "pcm_mulaw")` or `("g722", None)`.

**Example:**
```yaml
- type: codec
  noise_ratio: 0.4
```

---

### 7. Drop Frequencies (`drop_freq`)
Applies random notch filters to drop frequency bands.

**Configuration Signature:**
```yaml
- type: drop_freq
  args:
    noise_ratio: float
    drop_freq_low: float
    drop_freq_high: float
    drop_count_low: int
    drop_count_high: int
```

**Parameters:**

* **noise_ratio** - (*float*) Probability of applying this augmentation.
* **drop_freq_low** - (*float*) Min normalized frequency range (0-1).
* **drop_freq_high** - (*float*) Max normalized frequency range (0-1).
* **drop_count_low** - (*int*) Min number of notches to apply.
* **drop_count_high** - (*int*) Max number of notches to apply.

**Example:**
```yaml
- type: drop_freq
  noise_ratio: 0.5
  drop_freq_low: 0.0
  drop_freq_high: 0.5
  drop_count_low: 1
  drop_count_high: 3
```

---

### 8. Drop Chunk (`drop_chunk`)
Zeros out (or replaces with noise) random time segments.

**Configuration Signature:**
```yaml
- type: drop_chunk
  args:
    noise_ratio: float
    drop_length_low: int
    drop_length_high: int
    drop_count_low: int
    drop_count_high: int
    noise_factor: float
```

**Parameters:**

* **noise_ratio** - (*float*) Probability of applying this augmentation.
* **drop_length_low** - (*int*) Min length of chunks in samples.
* **drop_length_high** - (*int*) Max length of chunks in samples.
* **drop_count_low** - (*int*) Min number of chunks.
* **drop_count_high** - (*int*) Max number of chunks.
* **noise_factor** - (*float*) If > 0, fills chunk with random noise scaled by this factor.

**Example:**
```yaml
- type: drop_chunk
  noise_ratio: 0.5
  drop_length_low: 1000
  drop_length_high: 5000
  drop_count_low: 1
  drop_count_high: 4
```

---

### 9. Clipping (`do_clip`)
Clips signal amplitude to simulate saturation.

**Configuration Signature:**
```yaml
- type: do_clip
  args:
    noise_ratio: float
    clip_low: float
    clip_high: float
```

**Parameters:**

* **noise_ratio** - (*float*) Probability of applying this augmentation.
* **clip_low** - (*float*) Min clipping threshold.
* **clip_high** - (*float*) Max clipping threshold.

**Example:**
```yaml
- type: do_clip
  noise_ratio: 0.3
  clip_low: 0.5
  clip_high: 0.9
```
