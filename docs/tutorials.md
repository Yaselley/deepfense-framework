# Tutorial: Adding New Components

This guide walks you through adding custom components to DeepFense using the standard Base Classes.

## Prerequisites

All new components must be:
1.  Defined in a python file within the appropriate directory (e.g., `deepfense/models/frontends/`).
2.  Decorated with the appropriate `@register_*` decorator.
3.  Inherit from the appropriate **Base Class** (`BaseFrontend`, `BaseBackend`, `BaseLoss`) found in `deepfense.models.base_model`.
4.  (Optional but recommended) Imported in `deepfense/models/__init__.py` so they are available without extra imports.

---

## 1. Adding a New Frontend

**Goal**: Add a simple spectrogram frontend.

1.  Create `deepfense/models/frontends/spectrogram.py`.
2.  Implement the class inheriting from `BaseFrontend`.

```python
import torch
import torchaudio
from deepfense.utils.registry import register_frontend
from deepfense.models.base_model import BaseFrontend

@register_frontend("SpectrogramFrontend")
class SpectrogramFrontend(BaseFrontend):
    def __init__(self, config):
        """
        Args:
            config (dict): Arguments from the YAML 'frontend.args' section.
        """
        super().__init__(config)
        self.n_fft = config.get("n_fft", 512)
        self.transform = torchaudio.transforms.Spectrogram(n_fft=self.n_fft)

    def forward(self, x):
        # x shape: (Batch, Time)
        # Output shape: (Batch, Freq, Time)
        return self.transform(x)

    @property
    def output_dim(self):
        # Optional: Define output dimension logic if needed by backend
        # For Spectrogram, freq bins = n_fft // 2 + 1
        return self.n_fft // 2 + 1
```

3.  **Use it in Config**:
    ```yaml
    frontend:
      type: "SpectrogramFrontend"
      args:
        n_fft: 1024
    ```

---

## 2. Adding a New Backend

**Goal**: Add a simple LSTM backend.

1.  Create `deepfense/models/backends/lstm.py`.

```python
import torch
import torch.nn as nn
from deepfense.utils.registry import register_backend
from deepfense.models.base_model import BaseBackend

@register_backend("SimpleLSTM")
class SimpleLSTM(BaseBackend):
    def __init__(self, config):
        super().__init__(config)
        # input_dim is automatically populated from frontend if available, 
        # or provided in config.
        
        hidden_dim = config.get("hidden_dim", 128)
        self.lstm = nn.LSTM(self.input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, config.get("output_dim", 128))

    def forward(self, x):
        # x: (Batch, Channels, Time) -> Need (Batch, Time, Channels)
        if x.ndim == 3:
            x = x.transpose(1, 2) 
        
        # LSTM output
        out, (h_n, c_n) = self.lstm(x)
        
        # Take last state
        last_hidden = h_n[-1] 
        
        # Project to embedding
        embedding = self.fc(last_hidden)
        return embedding
```

---

## 3. Adding a New Unified Loss

**Goal**: Add a simple Binary Cross Entropy loss wrapper.

1.  Create `deepfense/models/losses/bce_loss.py`.

```python
import torch.nn as nn
from deepfense.utils.registry import register_loss
from deepfense.models.base_model import BaseLoss

@register_loss("BinaryCrossEntropy")
class BinaryCrossEntropy(BaseLoss):
    def __init__(self, config):
        super().__init__(config)
        # self.bonafide_label and self.spoof_label are available
        
        self.in_dim = config["embedding_dim"]
        
        # Projection to 1 output (0-1)
        self.fc = nn.Linear(self.in_dim, 1)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, embeddings, targets, logits=None):
        # targets are 0 or 1
        
        if logits is None:
            logits = self.get_logits(embeddings)
            
        # BCE expects float targets of shape (N, 1) or (N,)
        return self.criterion(logits.squeeze(), targets.float())

    def get_logits(self, embeddings):
        # Returns raw logits (before sigmoid)
        return self.fc(embeddings)

    def get_score(self, embeddings):
        # Return score for evaluation (e.g. probability of bonafide)
        logits = self.get_logits(embeddings).squeeze()
        # For metric calculation, we usually want a score where Higher = Bonafide.
        # If bonafide_label is 1, logits are fine (sigmoid(logits) -> P(1)).
        # If bonafide_label is 0, we might negate them.
        if self.bonafide_label == 1:
            return logits
        else:
            return -logits
```

---

## 4. Adding a New Metric

**Goal**: Add a custom accuracy metric.

1.  Create `deepfense/training/evaluations/my_metric.py`.

```python
from deepfense.utils.registry import register_metric
import numpy as np

@register_metric("MyCustomAcc")
def compute_my_acc(labels, scores, params):
    """
    labels: Ground truth (0 or 1)
    scores: 1D Scores (Higher = Bonafide)
    params: Dict from config
    """
    # Simple threshold at 0 (assuming LLR or Logits)
    preds = (scores > 0).astype(int)
    
    # If bonafide is 0, we might need to flip logic depending on how scores are defined
    # But typically scores > threshold => Bonafide (1).
    
    acc = (preds == labels).mean()
    return {"MyCustomAcc": acc}
```

2.  **Use it in Config**:
    ```yaml
    training:
      metrics:
        MyCustomAcc: {}
    ```
