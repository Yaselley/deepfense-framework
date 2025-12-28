# Adding a New Dataset

This guide shows you how to add a custom dataset class to DeepFense.

## Overview

Datasets in DeepFense load audio files and labels, applying transforms and returning samples. They must inherit from `BaseDataset` and be registered with `@register_dataset`. Datasets work with parquet files containing metadata about audio paths and labels.

## Step-by-Step Guide

### Step 1: Create the Dataset File

Create a new file `deepfense/data/my_dataset.py`:

```python
import torch
import pandas as pd
import numpy as np
import soundfile as sf
from pathlib import Path
from deepfense.data.base_dataset import BaseDataset
from deepfense.utils.registry import register_dataset


@register_dataset("MyCustomDataset")
class MyCustomDataset(BaseDataset):
    """
    Custom dataset for audio spoofing detection.
    
    Args:
        cfg: Dictionary containing configuration parameters
            - parquet_files: List of paths to parquet files
            - label_map: Dictionary mapping labels to integers
            - sampling_rate: Target sampling rate (default: 16000)
            - root_dir: Root directory for audio files (optional)
    """
    
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.parquet_files = cfg.get("parquet_files", [])
        self.label_map = cfg.get("label_map", {"bonafide": 1, "spoof": 0})
        self.sampling_rate = cfg.get("sampling_rate", 16000)
        self.root_dir = cfg.get("root_dir", None)
        
        # Load data
        self.samples = self._load_data()
    
    def _load_data(self):
        """Load data from parquet files."""
        all_samples = []
        
        for parquet_file in self.parquet_files:
            if not Path(parquet_file).exists():
                raise FileNotFoundError(f"Parquet file not found: {parquet_file}")
            
            df = pd.read_parquet(parquet_file)
            
            # Validate required columns
            required_cols = ["path", "label"]
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Missing required column '{col}' in {parquet_file}")
            
            # Add dataset name if available
            dataset_name = df.get("dataset_name", "unknown").iloc[0] if "dataset_name" in df.columns else "unknown"
            
            # Convert to list of dictionaries
            for _, row in df.iterrows():
                sample = {
                    "path": row["path"],
                    "label": row["label"],
                    "ID": row.get("ID", f"{dataset_name}_{len(all_samples)}"),
                    "dataset_name": dataset_name
                }
                all_samples.append(sample)
        
        return all_samples
    
    def __len__(self):
        """Return dataset size."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        sample = self.samples[idx]
        
        # Load and process audio
        audio = self._load_audio(sample["path"])
        label = self.label_map.get(sample["label"], 0)
        
        return {
            "x": torch.tensor(audio, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
            "ID": sample["ID"],
            "dataset_name": sample["dataset_name"]
        }
    
    def _load_audio(self, path):
        """Load audio from file."""
        # Handle relative paths
        if self.root_dir and not Path(path).is_absolute():
            path = Path(self.root_dir) / path
        else:
            path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        
        # Load audio
        audio, sr = sf.read(str(path))
        
        # Resample if needed
        if sr != self.sampling_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sampling_rate)
        
        # Ensure mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        
        return audio.astype(np.float32)
```

### Step 2: Register in __init__.py

Import your dataset in `deepfense/data/__init__.py`:

```python
from . import detection_dataset
from . import base_dataset
from . import my_dataset  # Add this line
```

**Important**: The import statement is required for the decorator to register the dataset when the module is loaded.

### Step 3: Create Parquet File Format

Your parquet file should have at least these columns:

```python
import pandas as pd

# Example: Creating a parquet file
data = {
    "ID": ["sample_001", "sample_002", "sample_003"],
    "path": [
        "/path/to/audio1.flac",
        "/path/to/audio2.flac",
        "/path/to/audio3.flac"
    ],
    "label": ["bonafide", "spoof", "bonafide"],
    "dataset_name": ["MyDataset", "MyDataset", "MyDataset"]  # Optional
}

df = pd.DataFrame(data)
df.to_parquet("my_dataset.parquet")
```

### Step 4: Use in Configuration

Use your dataset in a YAML configuration file:

```yaml
data:
  sampling_rate: 16000
  label_map:
    bonafide: 1
    spoof: 0

  train:
    dataset_type: "MyCustomDataset"  # Your registered name
    parquet_files: ["/path/to/train.parquet"]
    root_dir: "/path/to/audio/root"  # Optional, for relative paths
    label_map: ${data.label_map}
    sampling_rate: ${data.sampling_rate}
    batch_size: 32
    shuffle: True
```

### Step 5: Verify Registration

Check that your dataset is registered:

```bash
deepfense list --component-type datasets
```

Or programmatically:

```python
from deepfense.data import *  # Import all datasets
from deepfense.utils.registry import DATASET_REGISTRY

# Check if registered
if "MyCustomDataset" in DATASET_REGISTRY:
    print("Dataset registered successfully!")
    print("Available datasets:", DATASET_REGISTRY.list())
```

## Complete Example: Dataset with Transforms

Here's a complete example that applies transforms:

```python
import torch
import pandas as pd
import numpy as np
import soundfile as sf
from pathlib import Path
from deepfense.data.base_dataset import BaseDataset
from deepfense.utils.registry import register_dataset
from deepfense.data.transforms import build_transforms_pipeline


@register_dataset("TransformedDataset")
class TransformedDataset(BaseDataset):
    """
    Dataset with built-in transform pipeline.
    """
    
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.parquet_files = cfg.get("parquet_files", [])
        self.label_map = cfg.get("label_map", {"bonafide": 1, "spoof": 0})
        self.sampling_rate = cfg.get("sampling_rate", 16000)
        
        # Build transform pipeline
        self.base_transform = build_transforms_pipeline(
            cfg.get("base_transform", [])
        )
        self.augment_transform = build_transforms_pipeline(
            cfg.get("augment_transform", [])
        )
        
        # Load data
        self.samples = self._load_data()
    
    def _load_data(self):
        """Load data from parquet files."""
        all_samples = []
        for parquet_file in self.parquet_files:
            df = pd.read_parquet(parquet_file)
            all_samples.extend(df.to_dict('records'))
        return all_samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load audio
        audio = self._load_audio(sample["path"])
        
        # Apply base transforms (always applied)
        if self.base_transform:
            audio = self.base_transform(audio)
        
        # Apply augment transforms (probabilistic)
        if self.augment_transform:
            audio = self.augment_transform(audio)
        
        label = self.label_map.get(sample["label"], 0)
        
        return {
            "x": torch.tensor(audio, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
            "ID": sample.get("ID", str(idx)),
            "dataset_name": sample.get("dataset_name", "unknown")
        }
    
    def _load_audio(self, path):
        """Load audio from file."""
        audio, sr = sf.read(path)
        if sr != self.sampling_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sampling_rate)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio.astype(np.float32)
```

## Example: Dataset with Custom Metadata

```python
@register_dataset("MetadataDataset")
class MetadataDataset(BaseDataset):
    """
    Dataset that preserves additional metadata.
    """
    
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.parquet_files = cfg.get("parquet_files", [])
        self.label_map = cfg.get("label_map", {"bonafide": 1, "spoof": 0})
        self.sampling_rate = cfg.get("sampling_rate", 16000)
        self.samples = self._load_data()
    
    def _load_data(self):
        all_samples = []
        for parquet_file in self.parquet_files:
            df = pd.read_parquet(parquet_file)
            all_samples.extend(df.to_dict('records'))
        return all_samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio = self._load_audio(sample["path"])
        label = self.label_map.get(sample["label"], 0)
        
        # Return with metadata
        result = {
            "x": torch.tensor(audio, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
            "ID": sample.get("ID", str(idx)),
            "dataset_name": sample.get("dataset_name", "unknown")
        }
        
        # Add any additional metadata
        for key in ["speaker_id", "recording_device", "environment"]:
            if key in sample:
                result[key] = sample[key]
        
        return result
    
    def _load_audio(self, path):
        audio, sr = sf.read(path)
        if sr != self.sampling_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sampling_rate)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio.astype(np.float32)
```

## Key Points

1. **Inherit from BaseDataset**: Provides the base structure
2. **Use @register_dataset decorator**: Register with a unique string name
3. **Implement __len__()**: Return the number of samples
4. **Implement __getitem__()**: Return a dictionary with required keys: `x`, `label`, `ID`, `dataset_name`
5. **Handle audio loading**: Load audio files, resample if needed, convert to mono
6. **Support parquet format**: Load data from parquet files with expected columns
7. **Import in __init__.py**: Critical for registration

## Required Return Format

Your `__getitem__()` method must return a dictionary with at least:

```python
{
    "x": torch.Tensor,        # Audio waveform [Time] or [Channels, Time]
    "label": torch.Tensor,    # Label (0 or 1) - scalar
    "ID": str,                # Unique identifier
    "dataset_name": str       # Dataset identifier
}
```

## Testing Your Dataset

Test your dataset before using it in training:

```python
import pandas as pd
from deepfense.data.my_dataset import MyCustomDataset

# Create a test parquet file
test_data = pd.DataFrame({
    "ID": ["test_001", "test_002"],
    "path": ["/path/to/audio1.flac", "/path/to/audio2.flac"],
    "label": ["bonafide", "spoof"],
    "dataset_name": ["TestDataset", "TestDataset"]
})
test_data.to_parquet("/tmp/test.parquet")

# Create dataset instance
config = {
    "parquet_files": ["/tmp/test.parquet"],
    "label_map": {"bonafide": 1, "spoof": 0},
    "sampling_rate": 16000
}

dataset = MyCustomDataset(config)

# Test dataset
print(f"Dataset size: {len(dataset)}")

# Get a sample
sample = dataset[0]
print(f"Audio shape: {sample['x'].shape}")
print(f"Label: {sample['label']}")
print(f"ID: {sample['ID']}")

# Test DataLoader
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=2, shuffle=False)

for batch in loader:
    print(f"Batch audio shape: {batch['x'].shape}")  # [Batch, Time]
    print(f"Batch labels: {batch['label']}")  # [Batch]
    break
```

## Next Steps

- See [Adding a New Frontend](adding_frontends.md) for frontend creation
- See [Adding Augmentations](adding_augmentations.md) for data augmentation
- See [Training Guide](training_with_cli.md) for how to train with your custom dataset
- See [Configuration Reference](../05_configuration.md) for full config options

