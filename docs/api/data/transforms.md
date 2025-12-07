# Base Transforms

Located in `deepfense.data.transforms.transforms`.

## Pad / Trim
Handles length normalization.

**Config**:
```yaml
base_transform:
  - type: "pad"
    max_len: 64600
    pad_type: "repeat" # or "zero"
```

For stochastic augmentations, see [Augmentations](augmentations.md).

