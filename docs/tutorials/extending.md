# Extending DeepFense

DeepFense is designed to be easily extensible. This guide shows you how to add new components without modifying the core library code.

## 1. Adding a New Loss Function

To add a new loss function (e.g., `FocalLoss`), subclass `BaseLoss` and decorate it with `@register_loss`.

```python
# deepfense/models/losses/focal_loss.py
import torch
import torch.nn as nn
from deepfense.models.base_model import BaseLoss
from deepfense.utils.registry import register_loss

@register_loss("FocalLoss")
class FocalLoss(BaseLoss):
    def __init__(self, config):
        super().__init__(config)
        self.gamma = config.get("gamma", 2.0)
        self.alpha = config.get("alpha", 1.0)
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, embeddings, labels, logits=None):
        """
        args:
            embeddings: Tensor [B, D] (if your loss needs embeddings)
            labels: Tensor [B]
            logits: Tensor [B, C] (Optional, if backend/loss computed it)
        """
        # If logits aren't passed, we might need a projection layer here
        # For simplicity, assume logits are passed or computed
        
        logpt = -self.ce(logits, labels)
        pt = torch.exp(logpt)
        loss = self.alpha * ((1 - pt) ** self.gamma) * (-logpt)
        return loss.mean()
```

**Usage in Config:**
```yaml
loss:
  - type: "FocalLoss"
    gamma: 2.5
```

## 2. Adding a New Frontend

Subclass `BaseFrontend` and use `@register_frontend`.

```python
# deepfense/models/frontends/my_feature.py
import torch.nn as nn
from deepfense.models.base_model import BaseFrontend
from deepfense.utils.registry import register_frontend

@register_frontend("MyFeatureExtractor")
class MyFeatureExtractor(BaseFrontend):
    def __init__(self, config):
        super().__init__(config)
        self.n_mels = config.get("n_mels", 80)
        # Initialize your model/layers here

    def forward(self, x, mask=None):
        # x: [B, T] (Audio waveform)
        # mask: [B, T] (1=Valid, 0=Pad)
        
        # ... extract features ...
        # output should be [B, Channels, Freq, Time]
        return features
```

## 3. Adding a New Backend

Subclass `BaseBackend` and use `@register_backend`.

```python
# deepfense/models/backends/simple_cnn.py
import torch.nn as nn
from deepfense.models.base_model import BaseBackend
from deepfense.utils.registry import register_backend

@register_backend("SimpleCNN")
class SimpleCNN(BaseBackend):
    def __init__(self, config):
        super().__init__(config)
        input_dim = config.get("input_dim")
        self.conv = nn.Conv2d(1, 32, kernel_size=3)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, 128)

    def forward(self, x):
        # x: [B, C, F, T]
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x # Returns embedding [B, 128]
```

## 4. Adding Optimizers & Schedulers

### Adding an Optimizer
Use `@register_optimizer`.

```python
# deepfense/training/optimizers/custom_opt.py
from deepfense.utils.registry import register_optimizer
import torch.optim as optim

@register_optimizer("sgd_nesterov")
def SGDNesterov(params, config):
    lr = config.get("lr", 0.01)
    return optim.SGD(params, lr=lr, momentum=0.9, nesterov=True)
```

### Adding a Scheduler
Use `@register_scheduler`.

```python
# deepfense/training/schedulers/custom_sched.py
from deepfense.utils.registry import register_scheduler
import torch.optim.lr_scheduler as lr_scheduler

@register_scheduler("plateau")
def ReduceLROnPlateau(optimizer, config):
    patience = config.get("patience", 5)
    return lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience)
```

## 5. Adding Augmentations

Augmentations are standard callables or classes registered with `@register_transform`.

```python
from deepfense.utils.registry import register_transform

@register_transform("AddWhiteNoise")
class AddWhiteNoise:
    def __init__(self, noise_level=0.01):
        self.noise_level = noise_level

    def __call__(self, x):
        # x is numpy array [T]
        noise = np.random.normal(0, self.noise_level, x.shape)
        return x + noise
```
