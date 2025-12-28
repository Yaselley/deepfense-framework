# Extending DeepFense - Quick Reference

DeepFense uses a **Registry Pattern** that makes it easy to add custom components. This is a quick reference guide. For detailed step-by-step tutorials, see:

- **[Adding a New Backend](adding_backends.md)** - Complete guide with examples
- **[Adding a New Frontend](adding_frontends.md)** - Complete guide with examples
- **[Adding a New Loss](adding_losses.md)** - Complete guide with examples
- **[Adding a New Dataset](adding_datasets.md)** - Complete guide with examples
- **[Adding Augmentations](adding_augmentations.md)** - Complete guide with examples
- **[Adding Optimizers](adding_optimizers.md)** - Complete guide with examples
- **[Adding Metrics](adding_metrics.md)** - Complete guide with examples
- **[Adding Schedulers](adding_schedulers.md)** - Complete guide with examples

---

## How the Registry Works

Every component is registered with a decorator, then instantiated from YAML config:

```python
# 1. Register with decorator
@register_backend("MyBackend")
class MyBackend(nn.Module):
    ...

# 2. Use in YAML
backend:
  type: "MyBackend"    # ← Matches decorator name
  args:
    param1: value1
```

That's it. DeepFense automatically finds and builds your component.

---

## Adding a New Dataset

### Step 1: Create the File

`deepfense/data/my_dataset.py`:

```python
import torch
import pandas as pd
from deepfense.data.base_dataset import BaseDataset
from deepfense.utils.registry import register_dataset


@register_dataset("MyCustomDataset")
class MyCustomDataset(BaseDataset):
    """Custom dataset for specialized data loading."""
    
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.data_path = cfg["data_path"]
        self.label_map = cfg["label_map"]
        
        # Load your data
        self.samples = self._load_data()
    
    def _load_data(self):
        # Your custom loading logic
        df = pd.read_csv(self.data_path)
        return df.to_dict('records')
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load and process audio
        audio = self._load_audio(sample["path"])
        label = self.label_map[sample["label"]]
        
        return {
            "x": torch.tensor(audio, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
            "ID": sample.get("ID", str(idx))
        }
    
    def _load_audio(self, path):
        import soundfile as sf
        audio, sr = sf.read(path)
        return audio
```

### Step 2: Register Import

`deepfense/data/__init__.py`:
```python
from . import detection_dataset
from . import my_dataset  # Add this
```

### Step 3: Use in Config

```yaml
data:
  train:
    dataset_type: "MyCustomDataset"   # ← Your decorator name
    data_path: "./data/train.csv"
    label_map: {"real": 1, "fake": 0}
```

---

## Adding a New Frontend

See **[Adding a New Frontend](adding_frontends.md)** for complete guide.

### Step 1: Create the File

`deepfense/models/frontends/my_frontend.py`:

```python
import torch
import torch.nn as nn
from deepfense.models.base_model import BaseFrontend
from deepfense.utils.registry import register_frontend


@register_frontend("my_ssl_model")
class MySSLFrontend(BaseFrontend):
    """Custom SSL-based frontend."""
    
    def __init__(self, config):
        super().__init__(config)
        
        self.model_path = config.get("ckpt_path")
        self.freeze = config.get("freeze", True)
        self.output_dim = config.get("output_dim", 768)
        
        # Load your model
        self.encoder = self._load_model()
        
        if self.freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def _load_model(self):
        # Your model loading logic
        model = torch.load(self.model_path)
        return model
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Raw audio [Batch, Time]
            mask: Optional attention mask
            
        Returns:
            Features [Batch, Time', Dim]
        """
        features = self.encoder(x)
        return features
```

### Step 2: Register Import

`deepfense/models/frontends/__init__.py`:
```python
from . import wav2vec2
from . import wavlm
from . import my_frontend  # Add this
```

### Step 3: Use in Config

```yaml
model:
  frontend:
    type: "my_ssl_model"         # ← Your decorator name
    args:
      ckpt_path: "/path/to/model.pt"
      freeze: True
      output_dim: 768
```

---

---

## Quick Reference Examples

Below are minimal examples for quick reference. **For detailed guides, see the dedicated tutorials linked above.**

## Adding a New Backend

See **[Adding a New Backend](adding_backends.md)** for complete guide.

### Step 1: Create the File

`deepfense/models/backends/my_backend.py`:

```python
import torch.nn as nn
from deepfense.utils.registry import register_backend


@register_backend("MyBackend")
class MyBackend(nn.Module):
    """Custom backend classifier."""
    
    def __init__(self, config):
        super().__init__()
        
        input_dim = config["input_dim"]  # From frontend
        hidden_dim = config.get("hidden_dim", 256)
        output_dim = config.get("output_dim", 64)
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self._output_dim = output_dim
    
    def forward(self, x):
        """
        Args:
            x: Features [Batch, Time, Dim]
            
        Returns:
            Embeddings [Batch, output_dim]
        """
        # Pool over time
        x = x.mean(dim=1)
        
        return self.network(x)
    
    @property
    def output_dim(self):
        return self._output_dim
```

### Step 2: Register Import

`deepfense/models/backends/__init__.py`:
```python
from . import aasist
from . import mlp
from . import my_backend  # Add this
```

### Step 3: Use in Config

```yaml
model:
  backend:
    type: "MyBackend"           # ← Your decorator name
    args:
      input_dim: 768            # Must match frontend output
      hidden_dim: 512
      output_dim: 128
```

---

## Adding a New Loss

See **[Adding a New Loss](adding_losses.md)** for complete guide.

### Step 1: Create the File

`deepfense/models/losses/my_loss.py`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from deepfense.models.base_model import BaseLoss
from deepfense.utils.registry import register_loss


@register_loss("MyContrastiveLoss")
class MyContrastiveLoss(BaseLoss):
    """Custom contrastive loss."""
    
    def __init__(self, config):
        super().__init__(config)
        
        embedding_dim = config["embedding_dim"]  # From backend
        n_classes = config.get("n_classes", 2)
        self.temperature = config.get("temperature", 0.07)
        
        # Classification head
        self.classifier = nn.Linear(embedding_dim, n_classes)
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, embeddings, targets, logits=None):
        """
        Compute loss.
        
        Args:
            embeddings: [Batch, embedding_dim]
            targets: [Batch]
            
        Returns:
            Loss scalar
        """
        logits = self.classifier(embeddings)
        loss = self.ce_loss(logits, targets.long())
        return loss
    
    def get_score(self, embeddings):
        """Get scores for inference."""
        logits = self.classifier(embeddings)
        # Return bonafide score - spoof score
        return logits[:, 1] - logits[:, 0]
    
    def get_logits(self, embeddings):
        """Get raw logits."""
        return self.classifier(embeddings)
```

### Step 2: Register Import

`deepfense/models/losses/__init__.py`:
```python
from . import cross_entropy
from . import oc_softmax
from . import my_loss  # Add this
```

### Step 3: Use in Config

```yaml
model:
  loss:
    - type: "MyContrastiveLoss"    # ← Your decorator name
      weight: 1.0
      embedding_dim: 128           # Must match backend output
      n_classes: 2
      temperature: 0.1
```

---

## Adding a New Augmentation

See **[Adding Augmentations](adding_augmentations.md)** for complete guide.

### Step 1: Add to Augmentations File

`deepfense/data/transforms/augmentations.py`:

```python
from deepfense.utils.registry import register_transform
import torch


@register_transform("my_augmentation")
def my_custom_augmentation(waveform, sr, config):
    """
    Custom augmentation.
    
    Args:
        waveform: Audio tensor
        sr: Sample rate
        config: Dict with parameters
        
    Returns:
        Augmented waveform
    """
    noise_ratio = config.get("noise_ratio", 0.5)
    intensity = config.get("intensity", 0.1)
    
    # Skip with probability
    if torch.rand(1).item() > noise_ratio:
        return waveform
    
    # Your augmentation logic
    noise = torch.randn_like(waveform) * intensity
    augmented = waveform + noise
    
    return augmented
```

### Step 2: Use in Config

```yaml
data:
  train:
    augment_transform:
      - type: "my_augmentation"    # ← Your decorator name
        noise_ratio: 0.5
        intensity: 0.05
```

---

## Adding a New Optimizer

See **[Adding Optimizers](adding_optimizers.md)** for complete guide.

### Step 1: Register in Registry

`deepfense/training/optimizers/utils.py`:

```python
@register_optimizer("my_optimizer")
def build_my_optimizer(params, config):
    """Custom optimizer."""
    from torch.optim import SGD
    
    lr = config.get("lr", 0.001)
    momentum = config.get("momentum", 0.9)
    nesterov = config.get("nesterov", True)
    
    return SGD(params, lr=lr, momentum=momentum, nesterov=nesterov)
```

### Step 2: Use in Config

```yaml
training:
  optimizer:
    type: "my_optimizer"        # ← Your decorator name
    lr: 0.001
    momentum: 0.95
    nesterov: True
```

---

## Adding a New Metric

See **[Adding Metrics](adding_metrics.md)** for complete guide.

### Step 1: Add to Metrics File

`deepfense/training/evaluations/metrics.py`:

```python
from deepfense.utils.registry import register_metric
import numpy as np


@register_metric("AUC")
def compute_auc(scores, labels, **kwargs):
    """Compute Area Under ROC Curve."""
    from sklearn.metrics import roc_auc_score
    
    if scores.ndim == 2:
        scores = scores[:, 1]  # Use positive class
    
    return roc_auc_score(labels, scores)


@register_metric("Precision")
def compute_precision(scores, labels, threshold=0.0, **kwargs):
    """Compute Precision."""
    from sklearn.metrics import precision_score
    
    predictions = (scores > threshold).astype(int)
    return precision_score(labels, predictions)
```

### Step 2: Use in Config

```yaml
training:
  metrics:
    AUC: {}                     # ← Your decorator name
    Precision:
      threshold: 0.5
    EER: {}
```

---

## Quick Reference

| Component | Decorator | File Location | Config Key | Detailed Guide |
|-----------|-----------|---------------|------------|----------------|
| Dataset | `@register_dataset("Name")` | `deepfense/data/` | `data.train.dataset_type` | [Adding Datasets](adding_datasets.md) |
| Frontend | `@register_frontend("Name")` | `deepfense/models/frontends/` | `model.frontend.type` | [Adding Frontends](adding_frontends.md) |
| Backend | `@register_backend("Name")` | `deepfense/models/backends/` | `model.backend.type` | [Adding Backends](adding_backends.md) |
| Loss | `@register_loss("Name")` | `deepfense/models/losses/` | `model.loss[].type` | [Adding Losses](adding_losses.md) |
| Augmentation | `@register_transform("Name")` | `deepfense/data/transforms/augmentations.py` | `augment_transform[].type` | [Adding Augmentations](adding_augmentations.md) |
| Optimizer | `@register_optimizer("Name")` | `deepfense/training/optimizers/utils.py` | `training.optimizer.type` | [Adding Optimizers](adding_optimizers.md) |
| Scheduler | `@register_scheduler("Name")` | `deepfense/training/schedulers/utils.py` | `training.scheduler.type` | [Adding Schedulers](adding_schedulers.md) |
| Metric | `@register_metric("Name")` | `deepfense/training/evaluations/metrics.py` | `training.metrics.Name` | [Adding Metrics](adding_metrics.md) |

---

## Testing Your Component

Before using in training, test your component:

```python
import torch
from deepfense.models.backends.my_backend import MyBackend

# Test backend
config = {"input_dim": 768, "hidden_dim": 256, "output_dim": 64}
model = MyBackend(config)

dummy_input = torch.randn(4, 100, 768)  # [Batch, Time, Features]
output = model(dummy_input)
print(f"Output shape: {output.shape}")  # Should be [4, 64]
```

---

## Summary

1. **Create your class** with the appropriate base class
2. **Add the decorator** (`@register_xxx("YourName")`)
3. **Import in `__init__.py`**
4. **Use in YAML config** with `type: "YourName"`

That's all it takes to extend DeepFense!
