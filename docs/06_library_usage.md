# Using DeepFense as a Library

This tutorial shows how to use DeepFense as an installed Python library. Instead of running `train.py`, you can build experiments programmatically in your own scripts or notebooks.

---

## Installation

```bash
pip install deepfense
```

---

## Quick Example

```python
import deepfense
from deepfense.utils.registry import build_detector, build_frontend, build_backend, build_loss

# Build a complete detector
model = build_detector("ModularDetector", {
    "frontend": {"type": "wav2vec2", "args": {"source": "huggingface", "ckpt_path": "facebook/wav2vec2-base", "freeze": True}},
    "backend": {"type": "MLP", "args": {"input_dim": 768, "projection": [256, 64]}},
    "loss": [{"type": "CrossEntropy", "embedding_dim": 64, "n_classes": 2}]
})

# Run inference
import torch
audio = torch.randn(1, 64000)  # Fake audio
output = model(audio)
print(f"Score: {output['scores']}")
```

---

## 1. Building Models

### Build Individual Components

```python
from deepfense.utils.registry import build_frontend, build_backend, build_loss

# Frontend
frontend = build_frontend("wavlm", {
    "source": "huggingface",
    "ckpt_path": "microsoft/wavlm-base",
    "freeze": True
})

# Backend  
backend = build_backend("AASIST", {
    "input_dim": 768,
    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32]
})

# Loss
loss = build_loss("OCSoftmax", {
    "embedding_dim": 32,
    "w_posi": 0.9,
    "w_nega": 0.2,
    "alpha": 20.0
})
```

### Build Complete Detector

```python
from deepfense.utils.registry import build_detector

config = {
    "frontend": {
        "type": "wavlm",
        "args": {
            "source": "huggingface", 
            "ckpt_path": "microsoft/wavlm-base",
            "freeze": True
        }
    },
    "backend": {
        "type": "AASIST",
        "args": {"input_dim": 768}
    },
    "loss": [{
        "type": "OCSoftmax",
        "embedding_dim": 32,
        "w_posi": 0.9,
        "w_nega": 0.2,
        "alpha": 20.0
    }]
}

model = build_detector("ModularDetector", config)
model.to("cuda")
```

---

## 2. Loading Data

### Using StandardDataset

```python
from deepfense.data.detection_dataset import StandardDataset
from torch.utils.data import DataLoader

dataset_config = {
    "parquet_files": ["/path/to/train.parquet"],
    "root_dir": "/path/to/audio/root",
    "label_map": {"bonafide": 1, "spoof": 0},
    "base_transform": [
        {"type": "pad", "max_len": 64600, "pad_type": "repeat"}
    ]
}

dataset = StandardDataset(dataset_config)
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# Iterate
for batch in loader:
    audio = batch["x"]        # [B, T]
    labels = batch["label"]   # [B]
    ids = batch["ID"]         # List of IDs
    break
```

### Using build_dataloader

```python
from deepfense.data.data_utils import build_dataloader

config = {
    "dataset_type": "StandardDataset",
    "parquet_files": ["/path/to/data.parquet"],
    "root_dir": "/path/to/audio",
    "label_map": {"bonafide": 1, "spoof": 0},
    "batch_size": 32,
    "shuffle": True,
    "num_workers": 4,
    "base_transform": [
        {"type": "pad", "max_len": 64600}
    ]
}

loader = build_dataloader(config)
```

---

## 3. Training Loop

### Basic Training

```python
import torch
import torch.optim as optim
from deepfense.utils.registry import build_detector
from deepfense.data.data_utils import build_dataloader

# Build model
model = build_detector("ModularDetector", model_config)
model.to("cuda")

# Build data
train_loader = build_dataloader(train_config)
val_loader = build_dataloader(val_config)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(50):
    model.train()
    for batch in train_loader:
        audio = batch["x"].to("cuda")
        labels = batch["label"].to("cuda")
        
        optimizer.zero_grad()
        
        # Forward
        outputs = model(audio)
        loss = model.compute_loss(outputs, labels)
        
        # Backward
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            audio = batch["x"].to("cuda")
            outputs = model(audio)
            all_scores.append(outputs["scores"].cpu())
            all_labels.append(batch["label"])
    
    scores = torch.cat(all_scores).numpy()
    labels = torch.cat(all_labels).numpy()
    
    # Compute EER
    from deepfense.training.evaluations.metrics import compute_eer
    eer = compute_eer(scores, labels)
    print(f"Epoch {epoch+1}: EER = {eer:.2%}")
```

### Using StandardTrainer

```python
from deepfense.training.standard_trainer import StandardTrainer
from deepfense.utils.registry import build_detector
from deepfense.data.data_utils import build_dataloader

# Build components
model = build_detector("ModularDetector", model_config)
train_loader = build_dataloader(train_config)
val_loader = build_dataloader(val_config)

# Trainer config
training_config = {
    "epochs": 50,
    "device": "cuda",
    "output_dir": "./outputs/my_experiment",
    "optimizer": {"type": "adam", "lr": 1e-4},
    "scheduler": {"type": "cosine_annealing", "T_max": 50},
    "monitor_metric": "EER",
    "monitor_mode": "min",
    "metrics": {"EER": {}, "F1_SCORE": {}}
}

# Create trainer
trainer = StandardTrainer(
    model=model,
    data_loaders={"train": train_loader, "val": val_loader},
    config=training_config
)

# Train
trainer.train()
```

---

## 4. Inference

### Single File Inference

```python
import torch
import torchaudio
from deepfense.utils.registry import build_detector

# Load model
model = build_detector("ModularDetector", model_config)
checkpoint = torch.load("best_model.pth")
model.load_state_dict(checkpoint["model_state"])
model.eval()
model.to("cuda")

# Load audio
def load_audio(path, target_sr=16000, max_len=64600):
    audio, sr = torchaudio.load(path)
    if sr != target_sr:
        audio = torchaudio.transforms.Resample(sr, target_sr)(audio)
    audio = audio.mean(dim=0)  # Mono
    if len(audio) < max_len:
        audio = torch.nn.functional.pad(audio, (0, max_len - len(audio)))
    else:
        audio = audio[:max_len]
    return audio.unsqueeze(0)  # Add batch dim

# Inference
audio = load_audio("/path/to/test.flac").to("cuda")
with torch.no_grad():
    output = model(audio)
    score = output["scores"].item()

print(f"Score: {score:.4f}")
print(f"Prediction: {'bonafide' if score > 0 else 'spoof'}")
```

### Batch Inference

```python
import pandas as pd
from tqdm import tqdm

# Load test data
test_df = pd.read_parquet("test.parquet")
results = []

for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
    audio = load_audio(row["path"]).to("cuda")
    with torch.no_grad():
        output = model(audio)
        score = output["scores"].item()
    
    results.append({
        "ID": row["ID"],
        "score": score,
        "prediction": "bonafide" if score > 0 else "spoof"
    })

results_df = pd.DataFrame(results)
results_df.to_csv("predictions.csv", index=False)
```

---

## 5. Custom Experiments

### Comparing Frontends

```python
from deepfense.utils.registry import build_detector

frontends = [
    ("wav2vec2", {"source": "huggingface", "ckpt_path": "facebook/wav2vec2-base"}),
    ("wavlm", {"source": "huggingface", "ckpt_path": "microsoft/wavlm-base"}),
    ("hubert", {"source": "huggingface", "ckpt_path": "facebook/hubert-base-ls960"}),
]

results = {}
for name, frontend_args in frontends:
    config = {
        "frontend": {"type": name, "args": {**frontend_args, "freeze": True}},
        "backend": {"type": "MLP", "args": {"input_dim": 768, "projection": [256, 64]}},
        "loss": [{"type": "CrossEntropy", "embedding_dim": 64, "n_classes": 2}]
    }
    
    model = build_detector("ModularDetector", config)
    # ... train and evaluate ...
    results[name] = eer

print(results)
```

### Hyperparameter Search

```python
import itertools

learning_rates = [1e-3, 1e-4, 1e-5]
batch_sizes = [16, 32, 64]

for lr, bs in itertools.product(learning_rates, batch_sizes):
    print(f"Training with lr={lr}, batch_size={bs}")
    
    train_config["batch_size"] = bs
    training_config["optimizer"]["lr"] = lr
    
    loader = build_dataloader(train_config)
    trainer = StandardTrainer(model, {"train": loader, "val": val_loader}, training_config)
    trainer.train()
```

---

## 6. Available Registries

```python
from deepfense.utils.registry import (
    FRONTEND_REGISTRY,
    BACKEND_REGISTRY,
    LOSS_REGISTRY,
    DATASET_REGISTRY,
    TRANSFORM_REGISTRY,
    OPTIMIZER_REGISTRY,
    METRIC_REGISTRY
)

# List all available components
print("Frontends:", list(FRONTEND_REGISTRY.keys()))
print("Backends:", list(BACKEND_REGISTRY.keys()))
print("Losses:", list(LOSS_REGISTRY.keys()))
```

---

## 7. Extending from Library

You can register new components even when using DeepFense as a library:

```python
from deepfense.utils.registry import register_backend
import torch.nn as nn

@register_backend("MyCustomBackend")
class MyCustomBackend(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc = nn.Linear(config["input_dim"], config["output_dim"])
    
    def forward(self, x):
        return self.fc(x.mean(dim=1))

# Now use it
from deepfense.utils.registry import build_backend

backend = build_backend("MyCustomBackend", {"input_dim": 768, "output_dim": 64})
```

---

## API Quick Reference

| Function | Description |
|----------|-------------|
| `build_detector(name, config)` | Build complete model |
| `build_frontend(name, config)` | Build frontend only |
| `build_backend(name, config)` | Build backend only |
| `build_loss(name, config)` | Build loss module |
| `build_dataloader(config)` | Build DataLoader |
| `build_transforms_pipeline(config)` | Build transforms |

---

## Complete Minimal Example

```python
import torch
from deepfense.utils.registry import build_detector
from deepfense.data.data_utils import build_dataloader
from deepfense.training.evaluations.metrics import compute_eer

# 1. Build Model
model = build_detector("ModularDetector", {
    "frontend": {"type": "wav2vec2", "args": {"source": "huggingface", "ckpt_path": "facebook/wav2vec2-base", "freeze": True}},
    "backend": {"type": "MLP", "args": {"input_dim": 768, "projection": [128, 32]}},
    "loss": [{"type": "CrossEntropy", "embedding_dim": 32, "n_classes": 2}]
}).to("cuda")

# 2. Build Data
train_loader = build_dataloader({
    "dataset_type": "StandardDataset",
    "parquet_files": ["train.parquet"],
    "label_map": {"bonafide": 1, "spoof": 0},
    "batch_size": 32,
    "base_transform": [{"type": "pad", "max_len": 64600}]
})

# 3. Train
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(10):
    model.train()
    for batch in train_loader:
        x, y = batch["x"].to("cuda"), batch["label"].to("cuda")
        outputs = model(x)
        loss = model.compute_loss(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} done")

# 4. Save
torch.save({"model_state": model.state_dict()}, "model.pth")
```
