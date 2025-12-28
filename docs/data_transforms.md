# Data Transforms, Padding, and Cropping

This guide covers all data transformation options in DeepFense, including padding, cropping, resampling, and augmentations.

---

## Overview

DeepFense applies transforms to audio data in two stages:

1. **Base Transforms**: Always applied (train/val/test) - preprocessing like padding, resampling
2. **Augmentations**: Only during training (probabilistic) - data augmentation like noise, RIR, etc.

---

## Base Transforms

Base transforms are deterministic preprocessing steps applied to all data.

### 1. Audio Loading (`load_audio`)

**Purpose**: Load audio files from disk and perform initial preprocessing

**Parameters**:
- `target_sr` (int, default: 16000): Target sample rate (audio is resampled if needed)
- `mono` (bool, default: True): Convert to mono (averages channels if multi-channel)

**Example**:
```yaml
data:
  train:
    base_transform:
      - type: "load_audio"
        args:
          target_sr: 16000
          mono: True
```

**How to Check**: Audio is automatically resampled if file SR ≠ target_sr. Set `mono: False` to keep stereo.

---

### 2. Padding/Cropping (`pad`)

**Purpose**: Ensure all audio has the same length for batching

**Parameters**:
- `max_len` (int, required): Target length in samples
  - Example: `160000` = 10 seconds at 16kHz
  - Example: `64000` = 4 seconds at 16kHz
- `random_pad` (bool, default: False): 
  - `False`: Crop from start if audio > max_len
  - `True`: Randomly crop (random start position) if audio > max_len
- `pad_type` (str, default: "repeat"):
  - `"repeat"`: Repeat the waveform to fill length if audio < max_len
  - Other types: Currently only "repeat" is supported

**Example**:
```yaml
data:
  train:
    base_transform:
      - type: "pad"
        args:
          max_len: 160000        # 10 seconds at 16kHz
          random_pad: True       # Random crop if longer
          pad_type: "repeat"     # Repeat if shorter
```

**Common Lengths**:
- `160000` samples = 10 seconds @ 16kHz
- `64000` samples = 4 seconds @ 16kHz
- `32000` samples = 2 seconds @ 16kHz

**How to Change**:
```yaml
# For longer audio (e.g., 15 seconds)
max_len: 240000  # 15 * 16000

# For shorter audio (e.g., 5 seconds)
max_len: 80000   # 5 * 16000

# To always crop from start (no randomness)
random_pad: False
```

---

### 3. Random Crop (`RandomCrop`)

**Purpose**: Randomly crop audio to fixed length (alternative to `pad` with `random_pad: True`)

**Parameters**:
- `output_size` (int, required): Target length in samples

**Example**:
```yaml
data:
  train:
    base_transform:
      - type: "RandomCrop"
        args:
          output_size: 160000  # 10 seconds at 16kHz
```

**Note**: `RandomCrop` is similar to `pad` with `random_pad: True`, but doesn't pad short audio (truncates instead).

---

## Augmentation Transforms

Augmentations are probabilistic transforms applied only during training to improve robustness.

### 1. RawBoost

**Purpose**: Advanced audio augmentation suite (noise, filtering, etc.)

**Parameters**:
- `noise_ratio` (float, 0.0-1.0, default: 1.0): Probability of applying augmentation
- `algo` (int, default: 5): Algorithm variant (1-5)
- `*` (various): Additional RawBoost-specific parameters

**Example**:
```yaml
data:
  train:
    augment_transform:
      - type: "rawboost"
        args:
          noise_ratio: 0.5      # Apply 50% of the time
          algo: 5
```

---

### 2. Room Impulse Response (RIR)

**Purpose**: Simulate room acoustics using impulse responses

**Parameters**:
- `noise_ratio` (float, 0.0-1.0): Probability of applying
- `csv_file` (str, required): Path to CSV file with RIR paths

**Example**:
```yaml
data:
  train:
    augment_transform:
      - type: "rir"
        args:
          noise_ratio: 0.3
          csv_file: "/path/to/rir_files.csv"
```

---

### 3. Codec Compression

**Purpose**: Simulate audio codec compression artifacts

**Parameters**:
- `noise_ratio` (float, 0.0-1.0): Probability of applying
- `*` (various): Codec-specific parameters

**Example**:
```yaml
data:
  train:
    augment_transform:
      - type: "codec"
        args:
          noise_ratio: 0.2
```

---

### 4. Additive Noise

**Purpose**: Add Gaussian noise to audio

**Parameters**:
- `noise_ratio` (float, 0.0-1.0): Probability of applying
- `snr_range` (list, default: [5, 15]): Signal-to-noise ratio range [min, max] in dB

**Example**:
```yaml
data:
  train:
    augment_transform:
      - type: "add_noise"  # or "AdditiveNoise"
        args:
          noise_ratio: 0.3
          snr_range: [10, 20]  # SNR between 10-20 dB
```

---

### 5. Speed Perturbation

**Purpose**: Vary playback speed (time stretching)

**Parameters**:
- `noise_ratio` (float, 0.0-1.0): Probability of applying
- `speed_range` (list): Speed variation range [min, max]
  - Example: `[0.9, 1.1]` = 90% to 110% speed

**Example**:
```yaml
data:
  train:
    augment_transform:
      - type: "speed_perturb"
        args:
          noise_ratio: 0.5
          speed_range: [0.95, 1.05]
```

---

### 6. Other Augmentations

- `morph`: Audio morphing
- `add_babble`: Add babble noise
- `drop_freq`: Frequency dropout
- `drop_chunk`: Time dropout
- `do_clip`: Clipping augmentation

See [Augmentations Documentation](components/augmentations.md) for complete list.

---

## Complete Transform Pipeline Example

```yaml
data:
  sampling_rate: 16000  # Global sample rate
  
  train:
    dataset_type: "DetectionDataset"
    parquet_files: ["/path/to/train.parquet"]
    
    # Base transforms (always applied)
    base_transform:
      - type: "load_audio"
        args:
          target_sr: 16000
          mono: True
      
      - type: "pad"
        args:
          max_len: 160000        # 10 seconds
          random_pad: True       # Random crop if longer
          pad_type: "repeat"     # Repeat if shorter
    
    # Augmentations (probabilistic, training only)
    augment_transform:
      - type: "rawboost"
        args:
          noise_ratio: 0.5
          algo: 5
      
      - type: "rir"
        args:
          noise_ratio: 0.3
          csv_file: "/path/to/rir.csv"
      
      - type: "add_noise"
        args:
          noise_ratio: 0.2
          snr_range: [5, 15]
      
      - type: "speed_perturb"
        args:
          noise_ratio: 0.3
          speed_range: [0.9, 1.1]
  
  val:
    # Validation: only base transforms, no augmentations
    base_transform:
      - type: "load_audio"
        args:
          target_sr: 16000
          mono: True
      
      - type: "pad"
        args:
          max_len: 160000
          random_pad: False      # No random crop for validation
          pad_type: "repeat"
    
    augment_transform: []  # No augmentations
```

---

## Checking Current Transform Configuration

### Method 1: Inspect Config File

```bash
# View your config file
cat config/train.yaml | grep -A 20 "base_transform\|augment_transform"
```

### Method 2: Check Saved Config

After training, check the saved config:

```bash
cat outputs/your_experiment/config.yaml | grep -A 20 "base_transform\|augment_transform"
```

### Method 3: Programmatic Check

```python
import yaml

with open("config/train.yaml", "r") as f:
    config = yaml.safe_load(f)

# Check base transforms
print("Base Transforms:")
for transform in config["data"]["train"]["base_transform"]:
    print(f"  - {transform['type']}: {transform.get('args', {})}")

# Check augmentations
print("\nAugmentations:")
for aug in config["data"]["train"].get("augment_transform", []):
    print(f"  - {aug['type']}: {aug.get('args', {})}")
```

---

## Common Transform Scenarios

### Scenario 1: Fixed-Length Audio (10 seconds)

```yaml
base_transform:
  - type: "pad"
    args:
      max_len: 160000        # 10 seconds @ 16kHz
      random_pad: True       # Random crop if > 10s
      pad_type: "repeat"     # Repeat if < 10s
```

### Scenario 2: Variable-Length Audio (No Padding)

For variable-length batches, you can skip padding and handle it in collate function (advanced):

```yaml
base_transform:
  - type: "load_audio"
    args:
      target_sr: 16000
      mono: True
# No pad transform - handled in DataLoader
```

### Scenario 3: Shorter Audio (4 seconds)

```yaml
base_transform:
  - type: "pad"
    args:
      max_len: 64000         # 4 seconds @ 16kHz
      random_pad: False      # Always crop from start
      pad_type: "repeat"
```

### Scenario 4: Longer Audio (15 seconds)

```yaml
base_transform:
  - type: "pad"
    args:
      max_len: 240000        # 15 seconds @ 16kHz
      random_pad: True
      pad_type: "repeat"
```

### Scenario 5: Aggressive Augmentation

```yaml
augment_transform:
  - type: "rawboost"
    args:
      noise_ratio: 0.8       # Apply 80% of the time
  - type: "rir"
    args:
      noise_ratio: 0.6
  - type: "add_noise"
    args:
      noise_ratio: 0.5
      snr_range: [0, 10]     # Lower SNR = more noise
  - type: "speed_perturb"
    args:
      noise_ratio: 0.4
      speed_range: [0.85, 1.15]  # Wider range
```

### Scenario 6: Minimal Augmentation

```yaml
augment_transform:
  - type: "add_noise"
    args:
      noise_ratio: 0.2       # Apply 20% of the time
      snr_range: [15, 25]    # Higher SNR = less noise
```

---

## Transform Order

Transforms are applied in the order specified:

1. **Base transforms** are applied first (in order)
2. **Augmentations** are applied after base transforms (in order)
3. Each augmentation is applied independently with its `noise_ratio` probability

**Example**:
```yaml
base_transform:
  - type: "load_audio"   # 1. Load audio
  - type: "pad"          # 2. Pad/crop

augment_transform:
  - type: "rawboost"     # 3. Apply RawBoost (50% chance)
  - type: "add_noise"    # 4. Apply noise (30% chance, independent)
```

---

## Troubleshooting

### Issue: Audio length mismatch

**Problem**: "RuntimeError: Expected input batch_size (X) to match target batch_size (Y)"

**Solution**: Ensure all audio is padded to the same length:
```yaml
base_transform:
  - type: "pad"
    args:
      max_len: 160000  # Must match your target length
```

### Issue: Out of memory

**Problem**: GPU out of memory during training

**Solutions**:
1. Reduce `max_len` (shorter audio):
   ```yaml
   max_len: 80000  # 5 seconds instead of 10
   ```
2. Reduce `batch_size`
3. Reduce number of augmentations

### Issue: Augmentations not applying

**Problem**: Augmentations seem to have no effect

**Check**:
1. Verify `noise_ratio > 0.0`
2. Ensure augmentations are in `train` section, not `val`
3. Check that transform is registered: `deepfense list --component-type transforms`

### Issue: Audio quality degradation

**Problem**: Too much augmentation causing poor training

**Solution**: Reduce augmentation probabilities:
```yaml
augment_transform:
  - type: "rawboost"
    args:
      noise_ratio: 0.3  # Reduce from 0.5
  - type: "add_noise"
    args:
      noise_ratio: 0.1  # Reduce from 0.3
      snr_range: [15, 25]  # Increase SNR (less noise)
```

---

## Summary Table

| Transform | Type | Purpose | Key Parameters | When Applied |
|-----------|------|---------|----------------|--------------|
| `load_audio` | Base | Load & resample | `target_sr`, `mono` | Always |
| `pad` | Base | Pad/crop to fixed length | `max_len`, `random_pad`, `pad_type` | Always |
| `RandomCrop` | Base | Random crop | `output_size` | Always |
| `rawboost` | Aug | Advanced augmentation | `noise_ratio`, `algo` | Training (probabilistic) |
| `rir` | Aug | Room simulation | `noise_ratio`, `csv_file` | Training (probabilistic) |
| `codec` | Aug | Codec simulation | `noise_ratio` | Training (probabilistic) |
| `add_noise` | Aug | Add noise | `noise_ratio`, `snr_range` | Training (probabilistic) |
| `speed_perturb` | Aug | Speed variation | `noise_ratio`, `speed_range` | Training (probabilistic) |

---

## Next Steps

- See [Configuration Reference](05_configuration.md) for all parameters
- See [Augmentations Documentation](components/augmentations.md) for complete augmentation list
- See [Adding Augmentations](user_guide/adding_augmentations.md) to create custom transforms
- See [Data Preparation](README.md#data-preparation) in README for parquet format

