# Adding a New Augmentation

This guide shows you how to add a custom data augmentation transform to DeepFense.

## Overview

Augmentations in DeepFense modify audio waveforms during training to improve model robustness. They are registered with `@register_transform` and can be used in the augmentation pipeline. Augmentations are typically applied probabilistically based on a `noise_ratio` parameter.

## Step-by-Step Guide

### Step 1: Add to Augmentations File

Add your augmentation function to `deepfense/data/transforms/augmentations.py`:

```python
from deepfense.utils.registry import register_transform
import torch
import numpy as np


@register_transform("my_augmentation")
class MyAugmentation:
    """
    Custom augmentation transform.
    
    This augmentation should be registered as a class with __call__ method
    for consistency with other transforms.
    """
    
    def __init__(self, noise_ratio: float = 1.0, intensity: float = 0.1, **kwargs):
        """
        Args:
            noise_ratio: Probability of applying augmentation (0.0 to 1.0)
            intensity: Intensity parameter for the augmentation
            **kwargs: Additional parameters
        """
        self.noise_ratio = noise_ratio
        self.intensity = intensity
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply augmentation to audio waveform.
        
        Args:
            x: Audio waveform as numpy array [Time] or [Channels, Time]
            
        Returns:
            Augmented audio waveform (same shape as input)
        """
        # Skip augmentation based on probability
        if np.random.random() > self.noise_ratio:
            return x
        
        # Apply your augmentation logic
        # Example: Add gaussian noise
        noise = np.random.randn(*x.shape) * self.intensity
        augmented = x + noise
        
        # Ensure output is within valid range
        augmented = np.clip(augmented, -1.0, 1.0)
        
        return augmented
```

Alternatively, you can use a function-based approach:

```python
@register_transform("my_augmentation_func")
def my_custom_augmentation(x: np.ndarray, noise_ratio: float = 1.0, intensity: float = 0.1, **kwargs) -> np.ndarray:
    """
    Custom augmentation function.
    
    Args:
        x: Audio waveform [Time]
        noise_ratio: Probability of applying (0.0 to 1.0)
        intensity: Augmentation intensity
        **kwargs: Additional parameters
        
    Returns:
        Augmented waveform
    """
    if np.random.random() > noise_ratio:
        return x
    
    noise = np.random.randn(*x.shape) * intensity
    augmented = x + noise
    return np.clip(augmented, -1.0, 1.0)
```

**Note**: The class-based approach is preferred as it matches the pattern used by existing augmentations like `RIR`, `RawBoost`, and `Codec`.

### Step 2: Verify Registration

The augmentation is automatically registered when the module is imported. Check that it's registered:

```bash
deepfense list --component-type augmentations
```

Or programmatically:

```python
from deepfense.data.transforms import augmentations  # Import to register
from deepfense.utils.registry import TRANSFORM_REGISTRY

# Check if registered
if "my_augmentation" in TRANSFORM_REGISTRY:
    print("Augmentation registered successfully!")
    print("Available augmentations:", TRANSFORM_REGISTRY.list())
```

### Step 3: Use in Configuration

Use your augmentation in a YAML configuration file:

```yaml
data:
  train:
    augment_transform:
      - type: "my_augmentation"
        noise_ratio: 0.5
        intensity: 0.05
      - type: "AdditiveNoise"
        noise_ratio: 0.3
        snr_range: [10, 20]
```

Augmentations are applied in sequence according to the list order.

## Complete Example: Time Stretch Augmentation

Here's a complete example of a time stretch augmentation:

```python
from deepfense.utils.registry import register_transform
import numpy as np
import librosa


@register_transform("time_stretch")
class TimeStretch:
    """
    Time stretching augmentation using librosa.
    
    Stretches or compresses audio in time without changing pitch.
    """
    
    def __init__(self, noise_ratio: float = 1.0, rate_range: tuple = (0.8, 1.2), **kwargs):
        """
        Args:
            noise_ratio: Probability of applying augmentation
            rate_range: Tuple (min_rate, max_rate) for time stretch factor
        """
        self.noise_ratio = noise_ratio
        self.rate_range = rate_range
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply time stretch.
        
        Args:
            x: Audio waveform [Time]
            
        Returns:
            Time-stretched audio [Time']
        """
        if np.random.random() > self.noise_ratio:
            return x
        
        # Random stretch rate
        rate = np.random.uniform(self.rate_range[0], self.rate_range[1])
        
        # Apply time stretch (assuming 16kHz, adjust if needed)
        stretched = librosa.effects.time_stretch(x, rate=rate)
        
        # Trim or pad to original length
        target_len = len(x)
        if len(stretched) > target_len:
            stretched = stretched[:target_len]
        else:
            padded = np.zeros(target_len)
            padded[:len(stretched)] = stretched
            stretched = padded
        
        return stretched
```

## Example: Pitch Shift Augmentation

```python
from deepfense.utils.registry import register_transform
import numpy as np
import librosa


@register_transform("pitch_shift")
class PitchShift:
    """
    Pitch shifting augmentation.
    
    Changes pitch without changing duration.
    """
    
    def __init__(self, noise_ratio: float = 1.0, n_steps_range: tuple = (-2, 2), sr: int = 16000, **kwargs):
        """
        Args:
            noise_ratio: Probability of applying
            n_steps_range: Range of pitch shifts in semitones
            sr: Sample rate
        """
        self.noise_ratio = noise_ratio
        self.n_steps_range = n_steps_range
        self.sr = sr
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply pitch shift.
        """
        if np.random.random() > self.noise_ratio:
            return x
        
        n_steps = np.random.uniform(self.n_steps_range[0], self.n_steps_range[1])
        shifted = librosa.effects.pitch_shift(x, sr=self.sr, n_steps=n_steps)
        
        return shifted
```

## Example: Volume Perturbation

```python
from deepfense.utils.registry import register_transform
import numpy as np


@register_transform("volume_perturb")
class VolumePerturb:
    """
    Random volume scaling augmentation.
    """
    
    def __init__(self, noise_ratio: float = 1.0, gain_range: tuple = (-6.0, 6.0), **kwargs):
        """
        Args:
            noise_ratio: Probability of applying
            gain_range: Range of gain in dB
        """
        self.noise_ratio = noise_ratio
        self.gain_range = gain_range
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply volume perturbation.
        """
        if np.random.random() > self.noise_ratio:
            return x
        
        # Random gain in dB
        gain_db = np.random.uniform(self.gain_range[0], self.gain_range[1])
        
        # Convert dB to linear scale
        gain_linear = 10 ** (gain_db / 20.0)
        
        # Apply gain
        perturbed = x * gain_linear
        
        # Clip to prevent clipping artifacts
        return np.clip(perturbed, -1.0, 1.0)
```

## Example: SpecAugment-style Time Masking

```python
from deepfense.utils.registry import register_transform
import numpy as np


@register_transform("time_mask")
class TimeMask:
    """
    Time masking augmentation (similar to SpecAugment).
    
    Masks a contiguous time segment of the audio.
    """
    
    def __init__(self, noise_ratio: float = 1.0, max_mask_len: int = 2000, **kwargs):
        """
        Args:
            noise_ratio: Probability of applying
            max_mask_len: Maximum length of mask in samples
        """
        self.noise_ratio = noise_ratio
        self.max_mask_len = max_mask_len
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply time mask.
        """
        if np.random.random() > self.noise_ratio:
            return x
        
        # Random mask length
        mask_len = np.random.randint(0, min(self.max_mask_len, len(x)))
        
        if mask_len == 0:
            return x
        
        # Random start position
        start = np.random.randint(0, max(1, len(x) - mask_len))
        
        # Create masked version
        masked = x.copy()
        masked[start:start + mask_len] = 0.0
        
        return masked
```

## Key Points

1. **Use @register_transform decorator**: Register with a unique string name
2. **Class-based approach**: Preferred pattern with `__init__` and `__call__` methods
3. **Noise ratio**: Use `noise_ratio` parameter to control application probability
4. **Input/Output**: Accept numpy array `[Time]` or `[Channels, Time]`, return same shape
5. **Preserve shape**: Ensure output has compatible shape for downstream processing
6. **Clip values**: Keep audio values in valid range (typically [-1.0, 1.0])
7. **No import needed**: Augmentations are registered when the module is imported

## Function Signature

Your augmentation should follow this pattern:

**Class-based (Preferred):**
```python
@register_transform("my_aug")
class MyAug:
    def __init__(self, noise_ratio: float = 1.0, **kwargs):
        self.noise_ratio = noise_ratio
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.random.random() > self.noise_ratio:
            return x
        # Apply augmentation
        return augmented_x
```

**Function-based:**
```python
@register_transform("my_aug")
def my_aug(x: np.ndarray, noise_ratio: float = 1.0, **kwargs) -> np.ndarray:
    if np.random.random() > noise_ratio:
        return x
    # Apply augmentation
    return augmented_x
```

## Testing Your Augmentation

Test your augmentation before using it in training:

```python
import numpy as np
from deepfense.data.transforms import augmentations  # Import to register
from deepfense.utils.registry import build_transform

# Create augmentation instance
aug_config = {
    "type": "my_augmentation",
    "noise_ratio": 1.0,
    "intensity": 0.1
}

aug = build_transform(aug_config)

# Test on dummy audio
dummy_audio = np.random.randn(16000).astype(np.float32)  # 1 second at 16kHz
augmented = aug(dummy_audio)

print(f"Input shape: {dummy_audio.shape}")
print(f"Output shape: {augmented.shape}")
print(f"Input range: [{dummy_audio.min():.3f}, {dummy_audio.max():.3f}]")
print(f"Output range: [{augmented.min():.3f}, {augmented.max():.3f}]")

# Test with noise_ratio = 0 (should not apply)
aug_no_apply = build_transform({"type": "my_augmentation", "noise_ratio": 0.0})
result = aug_no_apply(dummy_audio)
assert np.allclose(result, dummy_audio), "Augmentation should not apply with noise_ratio=0"
```

## Combining Multiple Augmentations

You can chain multiple augmentations in your config:

```yaml
data:
  train:
    augment_transform:
      - type: "time_stretch"
        noise_ratio: 0.5
        rate_range: [0.9, 1.1]
      - type: "pitch_shift"
        noise_ratio: 0.5
        n_steps_range: [-1, 1]
      - type: "volume_perturb"
        noise_ratio: 0.3
        gain_range: [-3.0, 3.0]
      - type: "AdditiveNoise"
        noise_ratio: 0.2
        snr_range: [15, 25]
```

Augmentations are applied sequentially, each with its own `noise_ratio` probability.

## Next Steps

- See [Adding a New Dataset](adding_datasets.md) for dataset creation
- See [Training Guide](training_with_cli.md) for how to use augmentations in training
- See [Configuration Reference](../05_configuration.md) for full config options
- See existing augmentations in `deepfense/data/transforms/augmentations.py` for reference

