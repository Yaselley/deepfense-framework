# Configuration Guide

DeepFense uses **OmegaConf** (YAML) for configuration. The config file is hierarchical and split into three main sections: `data`, `model`, and `training`.

## 1. Data Section (`data`)

Controls dataset loading and processing.

```yaml
data:
  sampling_rate: 16000
  label_map: {"bonafide": 1, "spoof": 0} # Global label map

  train:
    dataset_type: "StandardDataset"
    dataset_names: ["ASVSpoof19"]
    parquet_files: ["/path/to/train.parquet"]
    batch_size: 32
    shuffle: True
    
    # Augmentations
    base_transform:
      - type: "pad"
        max_len: 64000
        
    augment_transform:
      # Use AugmentationPipeline for advanced control
      - type: "augmentation_pipeline"
        mode: "sequential" # or "parallel"
        k: 2               # Apply 2 random transforms from the list (if sequential)
        p: 1.0             # Probability to apply pipeline
        transforms:
          - type: "rawboost"
            noise_ratio: 1.0 # Always apply if selected by pipeline
            algo: 5
          - type: "rir"
            noise_ratio: 1.0
            csv_file: "/path/to/rirs.csv"

  val:
    dataset_type: "StandardDataset"
    # ... similar structure ...
```

## 2. Model Section (`model`)

Defines the detector architecture.

```yaml
model:
  type: "StandardDetector"

  frontend:
    type: "wavlm"
    args:
      ckpt_path: "/path/to/model.pt"

  backend:
    type: "AASIST"
    args:
      filts: [ [1, 32], [32, 32], ... ]
      gat_dims: [64, 32]
      # ... specific backend args ...
  
  # Or use MLP
  # backend:
  #   type: "MLP"
  #   args:
  #     projection: [512, 256]
  #     output_dim: 128

  loss:
    # List of losses (Unified Loss Modules)
    - type: "CrossEntropy"
      weight: 1.0
      embedding_dim: 128 # Must match backend output
      n_classes: 2
```

## 3. Training Section (`training`)

Controls the training loop, optimizer, and evaluation.

```yaml
training:
  trainer: "StandardTrainer"
  output_dir: "./outputs/"
  epochs: 50
  device: "cuda"
  
  # Validation frequency
  eval_every_steps: 1000
  monitor_metric: "EER"  # Metric to track for best checkpoint
  monitor_mode: "min"    # "min" or "max"

  optimizer:
    type: "adam"
    lr: 0.0001
    weight_decay: 0.0001

  scheduler:
    type: "cosine"
    T_max: 50

  metrics:
    EER: {}
    minDCF:
      Pspoof: 0.05
      Cmiss: 1
      Cfa: 1
```
