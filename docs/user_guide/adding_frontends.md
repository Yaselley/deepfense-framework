# Adding a New Frontend

This guide shows you how to add a custom frontend (feature extractor) to DeepFense.

## Overview

Frontends in DeepFense extract features from raw audio. They typically wrap pre-trained self-supervised learning models like Wav2Vec2, WavLM, or HuBERT. Frontends must inherit from `BaseFrontend` and be registered with `@register_frontend`.

## Step-by-Step Guide

### Step 1: Create the Frontend File

Create a new file `deepfense/models/frontends/my_frontend.py`:

```python
import torch
import torch.nn as nn
from deepfense.models.base_model import BaseFrontend
from deepfense.utils.registry import register_frontend


@register_frontend("my_ssl_model")
class MySSLWrapper(BaseFrontend):
    """
    Custom SSL model wrapper for audio feature extraction.
    
    Args:
        config: Dictionary containing configuration parameters
            - ckpt_path: Path to pretrained model checkpoint (optional)
            - freeze: Whether to freeze the model parameters (default: True)
            - output_dim: Output feature dimension
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        self.model_path = config.get("ckpt_path", None)
        self.freeze = config.get("freeze", True)
        self.output_dim = config.get("output_dim", 768)
        
        # Load your model
        self.encoder = self._load_model()
        
        if self.freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def _load_model(self):
        """Load the pretrained model."""
        # Example: Load from transformers
        from transformers import AutoModel
        
        model_name = self.config.get("model_name", "facebook/wav2vec2-base")
        model = AutoModel.from_pretrained(model_name)
        
        # Load checkpoint if provided
        if self.model_path:
            checkpoint = torch.load(self.model_path, map_location="cpu")
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
        
        return model
    
    def forward(self, x, mask=None):
        """
        Forward pass.
        
        Args:
            x: Raw audio [Batch, Time]
            mask: Optional attention mask [Batch, Time]
            
        Returns:
            Features [Batch, Time', Dim]
        """
        # Get model outputs
        outputs = self.encoder(x, attention_mask=mask)
        
        # Extract features (adjust based on your model)
        features = outputs.last_hidden_state  # [Batch, Time, Dim]
        
        return features
    
    @property
    def feature_dim(self):
        """Return feature dimension."""
        return self.output_dim
```

### Step 2: Register in __init__.py

Import your frontend in `deepfense/models/frontends/__init__.py`:

```python
from .wav2vec2 import Wav2VecWrapper
from .wavlm import WavLMWrapper
from .hubert import HubertWrapper
from .mert import MERTWrapper
from .eat import EATWrapper
from .my_frontend import MySSLWrapper  # Add this line
```

**Important**: The import statement is required for the decorator to register the frontend.

### Step 3: Use in Configuration

Use your frontend in a YAML configuration file:

```yaml
model:
  type: "Detector"
  frontend:
    type: "my_ssl_model"  # Your registered name
    args:
      ckpt_path: "/path/to/checkpoint.pt"  # Optional
      freeze: True
      output_dim: 768
      model_name: "facebook/wav2vec2-base"  # If using transformers
```

### Step 4: Verify Registration

Check that your frontend is registered:

```bash
deepfense list --component-type frontends
```

Or programmatically:

```python
from deepfense.models.frontends import *  # Import all frontends
from deepfense.utils.registry import FRONTEND_REGISTRY

# Check if registered
if "my_ssl_model" in FRONTEND_REGISTRY:
    print("Frontend registered successfully!")
    print("Available frontends:", FRONTEND_REGISTRY.list())
```

## Complete Example: Custom Wav2Vec2 Variant

Here's a complete example for a custom Wav2Vec2 variant:

```python
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from deepfense.models.base_model import BaseFrontend
from deepfense.utils.registry import register_frontend


@register_frontend("custom_wav2vec2")
class CustomWav2Vec2Wrapper(BaseFrontend):
    """
    Custom Wav2Vec2 wrapper with layer selection.
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        self.model_path = config.get("ckpt_path", None)
        self.freeze = config.get("freeze", True)
        self.layer_index = config.get("layer_index", -1)  # Use last layer
        self.output_dim = config.get("output_dim", 768)
        
        self.model = self._load_model()
        
        if self.freeze:
            for param in self.model.parameters():
                param.requires_grad = False
    
    def _load_model(self):
        """Load Wav2Vec2 model."""
        model_name = self.config.get("model_name", "facebook/wav2vec2-base")
        model = Wav2Vec2Model.from_pretrained(model_name)
        
        if self.model_path:
            state_dict = torch.load(self.model_path, map_location="cpu")
            model.load_state_dict(state_dict)
        
        return model
    
    def forward(self, x, mask=None):
        """
        Args:
            x: [Batch, Time] - raw audio
            mask: [Batch, Time] - attention mask
        Returns:
            [Batch, Time', Dim] - features
        """
        # Wav2Vec2 expects normalized input
        # You may need to normalize x first
        
        outputs = self.model(x, attention_mask=mask)
        
        # Get features from specific layer if needed
        if hasattr(outputs, 'hidden_states') and self.layer_index >= 0:
            features = outputs.hidden_states[self.layer_index]
        else:
            features = outputs.last_hidden_state
        
        return features
    
    @property
    def feature_dim(self):
        return self.output_dim
```

## Example: Feature Extraction from Custom Model

If you have a custom model architecture:

```python
import torch
import torch.nn as nn
from deepfense.models.base_model import BaseFrontend
from deepfense.utils.registry import register_frontend


@register_frontend("custom_cnn")
class CustomCNNWrapper(BaseFrontend):
    """
    Custom CNN-based feature extractor.
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        self.freeze = config.get("freeze", False)
        self.output_dim = config.get("output_dim", 512)
        
        # Build CNN encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, self.output_dim, kernel_size=3, stride=2, padding=1),
        )
        
        if self.freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def forward(self, x, mask=None):
        """
        Args:
            x: [Batch, Time] - raw audio
            mask: Not used in this example
        Returns:
            [Batch, Time', Dim] - features
        """
        # Add channel dimension: [Batch, Time] -> [Batch, 1, Time]
        x = x.unsqueeze(1)
        
        # CNN forward: [Batch, 1, Time] -> [Batch, Dim, Time']
        features = self.encoder(x)
        
        # Transpose: [Batch, Dim, Time'] -> [Batch, Time', Dim]
        features = features.transpose(1, 2)
        
        return features
    
    @property
    def feature_dim(self):
        return self.output_dim
```

## Key Points

1. **Inherit from BaseFrontend**: Provides the base interface
2. **Use @register_frontend decorator**: Register with a unique string name
3. **Implement forward()**: Must accept [Batch, Time] audio and return [Batch, Time', Dim] features
4. **Handle masking**: Support optional attention masks for variable-length sequences
5. **Set feature_dim property**: Required so backends know input dimension
6. **Import in __init__.py**: Critical for registration
7. **Freeze parameters**: Typically freeze pretrained models during training

## Testing Your Frontend

Test your frontend before using it in training:

```python
import torch
from deepfense.models.frontends.my_frontend import MySSLWrapper

# Create frontend instance
config = {
    "ckpt_path": None,  # Or path to checkpoint
    "freeze": True,
    "output_dim": 768
}
frontend = MySSLWrapper(config)

# Test forward pass
dummy_audio = torch.randn(2, 16000)  # [Batch=2, Time=16000] - 1 second at 16kHz
features = frontend(dummy_audio)
print(f"Input shape: {dummy_audio.shape}")
print(f"Output shape: {features.shape}")  # Should be [2, Time', 768]

# Test with mask
mask = torch.ones(2, 16000)
mask[0, 8000:] = 0  # Second half masked for first sample
features_masked = frontend(dummy_audio, mask=mask)
print(f"Features with mask shape: {features_masked.shape}")
```

## Next Steps

- See [Adding a New Backend](adding_backends.md) for backend creation
- See [Training Guide](training.md) for how to train with your custom frontend
- See [Configuration Reference](../05_configuration.md) for full config options

