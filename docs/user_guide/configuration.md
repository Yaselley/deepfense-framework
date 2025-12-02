# Configuration Guide

DeepFense uses a hierarchical YAML configuration system. The main config file (usually `config/train.yaml`) controls every aspect of the experiment.

## Structure

### 1. Global Settings
```yaml
exp_name: "MyExperiment"  # Folder name for outputs
output_dir: "./outputs/"
seed: 1234                # Random seed for reproducibility
```

### 2. Data Configuration (`data`)
Defines training, validation, and testing datasets.

```yaml
data:
  sampling_rate: 16000
  label_map: {"bonafide": 1, "spoof": 0} # Maps string labels to integers

  train:
    dataset_type: "StandardDataset"
    dataset_names: ["ASVSpoof19"]
    parquet_files: ["/path/to/train.parquet"]
    batch_size: 32
    shuffle: True
    
    # Augmentation Pipeline
    augment_transform:
      - type: "rawboost"
        noise_ratio: 0.5
        algo: 5
      - type: "rir"
        noise_ratio: 0.3
        csv_file: "rirs.csv"
```

### 3. Model Configuration (`model`)
Selects the architecture components.

```yaml
model:
  type: "StandardDetector"

  frontend:
    type: "wav2vec2"
    args:
      source: "huggingface"
      ckpt_path: "facebook/wav2vec2-base"
      freeze: True

  backend:
    type: "AASIST"
    args:
      filts: [70, [1, 32], [32, 32], [32, 64], [64, 64]]
      gat_dims: [64, 32]

  loss:
    - type: "OCSoftmax" # Primary Loss
      weight: 1.0
      embedding_dim: 32 # Must match backend output
      # ... specific loss args ...
```

### 4. Training Configuration (`training`)
Controls the loop, optimizer, and scheduler.

```yaml
training:
  trainer: "StandardTrainer"
  epochs: 50
  device: "cuda"
  
  # Checkpointing
  monitor_metric: "EER" # Metric to track for 'best_model.pth'
  monitor_mode: "min"   # 'min' for EER/Loss, 'max' for Accuracy
  
  optimizer:
    type: "adam"
    lr: 0.0001
    weight_decay: 0.0001
    
  scheduler:
    type: "cosine_annealing"
    T_max: 50
```

