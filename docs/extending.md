# Extending DeepFense

DeepFense is designed to be modular and easily extensible. To add new components, you should inherit from the provided base classes and register your component.

## Base Classes location
`deepfense.models.base_model`

## 1. Adding a New Loss
To add a new loss function (which may include a projection/mapper layer), inherit from `BaseLoss`.

```python
from deepfense.models.base_model import BaseLoss
from deepfense.utils.registry import register_loss
import torch

@register_loss("MyNewLoss")
class MyNewLoss(BaseLoss):
    def __init__(self, config):
        super().__init__(config)
        # Initialize your layers (e.g. linear projection) or loss function
        self.fc = torch.nn.Linear(config["embedding_dim"], config["n_classes"])
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, embeddings, targets, logits=None):
        """
        Compute the training loss.
        """
        if logits is None:
            logits = self.fc(embeddings)
        return self.criterion(logits, targets)

    def get_score(self, embeddings):
        """
        Compute the scores used for evaluation (EER, minDCF, etc.).
        Should return logits or similarity scores.
        """
        return self.fc(embeddings)
```

## 2. Adding a New Frontend
Frontends process raw audio waveforms into features. Inherit from `BaseFrontend`.

```python
from deepfense.models.base_model import BaseFrontend
from deepfense.utils.registry import register_frontend
import torch

@register_frontend("MyFrontend")
class MyFrontend(BaseFrontend):
    def __init__(self, config):
        super().__init__(config)
        # Initialize model
        
    def forward(self, x):
        # x: [Batch, Time]
        # return: [Batch, Time, Channels]
        return features
```

## 3. Adding a New Backend
Backends process features into fixed-size embeddings. Inherit from `BaseBackend`.

```python
from deepfense.models.base_model import BaseBackend
from deepfense.utils.registry import register_backend
import torch

@register_backend("MyBackend")
class MyBackend(BaseBackend):
    def __init__(self, config):
        super().__init__(config)
        # Initialize model
        
    def forward(self, x):
        # x: [Batch, Time, Channels]
        # return: [Batch, Embedding_Dim]
        return embeddings
```

## 4. Adding Optimizers and Schedulers
These typically wrap PyTorch classes. 

```python
from deepfense.utils.registry import register_optimizer, register_scheduler
import torch.optim as optim

@register_optimizer("my_adam")
def build_my_adam(params, config):
    return optim.Adam(params, lr=config.get("lr", 1e-3))
```

