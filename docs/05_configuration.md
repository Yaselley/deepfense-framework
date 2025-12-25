# Configuration Reference

Complete reference for all YAML configuration parameters in DeepFense.

---

## Configuration Structure

```yaml
# Top-level structure
exp_name: string      # Experiment name
output_dir: string    # Output directory
seed: integer         # Random seed

data: {...}           # Data configuration
model: {...}          # Model configuration
training: {...}       # Training configuration
```

---

## Global Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `exp_name` | string | `"default_exp"` | Name for the experiment folder |
| `output_dir` | string | `"./outputs/"` | Base directory for outputs |
| `seed` | integer | `42` | Random seed for reproducibility |

---

## Data Configuration

### Structure

```yaml
data:
  sampling_rate: 16000
  label_map:
    bonafide: 1
    spoof: 0
  
  train: {...}    # Training data config
  val: {...}      # Validation data config
  test: {...}     # Test data config (optional)
```

### Dataset Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset_type` | string | `"StandardDataset"` | Dataset class name |
| `parquet_files` | list[string] | **required** | Paths to Parquet files |
| `dataset_names` | list[string] | `null` | Names for each Parquet file |
| `batch_size` | integer | `32` | Batch size |
| `shuffle` | boolean | `true` | Shuffle data |
| `num_workers` | integer | `4` | DataLoader workers |
| `drop_last` | boolean | `false` | Drop incomplete batches |

### Base Transform

Applied to all data (train, val, test):

```yaml
base_transform:
  - type: load_audio
    target_sr: 16000
    mono: true
  
  - type: pad
    max_len: 64600
    random_pad: true     # Random crop if audio > max_len
    pad_type: repeat     # How to pad short audio
```

| Transform | Parameters | Description |
|-----------|------------|-------------|
| `load_audio` | `target_sr`, `mono` | Load audio file |
| `pad` | `max_len`, `random_pad`, `pad_type` | Pad/truncate to fixed length |

### Augmentation Transform (Training Only)

```yaml
augment_transform:
  - type: augmentation_pipeline
    p: 0.5                    # Probability of applying
    mode: parallel            # Selection mode
    execution: chain          # Execution mode
    concat_original: false    # Keep original?
    transforms:
      - type: rawboost
        noise_ratio: 1.0
        algo: 5
      - type: rir
        noise_ratio: 0.8
        csv_file: /path/to/rirs.csv
```

#### Pipeline Modes

| Mode | Execution | Result |
|------|-----------|--------|
| `parallel` | `chain` | Pick ONE transform randomly |
| `sequential` | `chain` | Apply ALL transforms in order |
| `sequential` | `independent` | Create separate copies with each transform |

#### Available Augmentations

| Type | Key Parameters |
|------|----------------|
| `rawboost` | `algo` (0-8), `noise_ratio` |
| `rir` | `csv_file`, `noise_ratio` |
| `add_noise` | `csv_file`, `snr_low`, `snr_high`, `noise_ratio` |
| `add_babble` | `csv_file`, `speaker_count`, `snr_low`, `snr_high` |
| `speed_perturb` | `speeds` (list), `noise_ratio` |
| `codec` | `noise_ratio` |
| `drop_freq` | `drop_freq_low`, `drop_freq_high`, `drop_count_low`, `drop_count_high` |
| `drop_chunk` | `drop_length_low`, `drop_length_high`, `drop_count_low`, `drop_count_high` |

---

## Model Configuration

### Structure

```yaml
model:
  type: "ModularDetector"
  
  frontend: {...}
  backend: {...}
  loss: [...]
```

### Frontend

```yaml
frontend:
  type: "wavlm"         # Frontend type
  args:
    source: "huggingface"
    ckpt_path: "microsoft/wavlm-base"
    freeze: true
```

| Frontend | Type Key | Key Args |
|----------|----------|----------|
| Wav2Vec2 | `wav2vec2` | `source`, `ckpt_path`, `freeze` |
| WavLM | `wavlm` | `source`, `ckpt_path`, `freeze` |
| HuBERT | `hubert` | `source`, `ckpt_path`, `freeze` |
| MERT | `mert` | `ckpt_path`, `freeze`, `trust_remote_code` |
| EAT | `eat` | `ckpt_path`, `freeze`, `trust_remote_code` |

### Backend

```yaml
backend:
  type: "AASIST"
  args:
    input_dim: 768       # Must match frontend output
    filts: [70, [1, 32], [32, 32], [32, 64], [64, 64]]
    gat_dims: [64, 32]
    pool_ratios: [0.5, 0.5, 0.5, 0.5]
    temperatures: [2.0, 2.0, 100.0, 100.0]
```

| Backend | Type Key | Key Args |
|---------|----------|----------|
| AASIST | `AASIST` | `input_dim`, `filts`, `gat_dims` |
| ECAPA-TDNN | `ECAPA_TDNN` | `channels`, `emb_dim` |
| RawNet2 | `RawNet2` | `filts`, `gru_node`, `emb_dim` |
| MLP | `MLP` | `input_dim`, `projection`, `pooling_type` |
| Nes2Net | `Nes2Net` | `strides`, `filts` |

### Loss Functions

Single loss:
```yaml
loss:
  type: "OCSoftmax"
  weight: 1.0
  embedding_dim: 32
  w_posi: 0.9
  w_nega: 0.2
  alpha: 20.0
```

Multiple losses:
```yaml
loss:
  - type: "OCSoftmax"
    weight: 1.0
    embedding_dim: 32
    w_posi: 0.9
    w_nega: 0.2
    alpha: 20.0
  
  - type: "CrossEntropy"
    weight: 0.5
    embedding_dim: 32
    n_classes: 2
```

| Loss | Type Key | Key Args |
|------|----------|----------|
| OC-Softmax | `OCSoftmax` | `embedding_dim`, `w_posi`, `w_nega`, `alpha` |
| AM-Softmax | `AMSoftmax` | `embedding_dim`, `n_classes`, `m`, `s` |
| A-Softmax | `ASoftmax` | `embedding_dim`, `n_classes`, `m` |
| Cross Entropy | `CrossEntropy` | `embedding_dim`, `n_classes` |

---

## Training Configuration

### Structure

```yaml
training:
  trainer: "StandardTrainer"
  device: "cuda"
  
  # Loop settings
  epochs: 50
  gradient_accumulation_steps: 1
  max_grad_norm: 1.0
  
  # Logging
  batch_log_interval: 50
  
  # Evaluation
  eval_every_epochs: 1
  metrics: ["EER", "F1"]
  
  # Checkpointing
  monitor_metric: "EER"
  monitor_mode: "min"
  save_every_epochs: 5
  early_stopping_patience: 10
  
  # Optimizer
  optimizer: {...}
  
  # Scheduler
  scheduler: {...}
  
  # WandB (optional)
  wandb: {...}
```

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `trainer` | string | `"StandardTrainer"` | Trainer class |
| `device` | string | `"cuda"` | Device (`cuda`, `cpu`, `cuda:0`) |
| `epochs` | integer | `50` | Total training epochs |
| `gradient_accumulation_steps` | integer | `1` | Accumulate gradients |
| `max_grad_norm` | float | `1.0` | Gradient clipping norm |

### Logging

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_log_interval` | integer | `50` | Log every N batches |

### Evaluation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eval_every_epochs` | integer | `1` | Evaluate every N epochs |
| `eval_every_steps` | integer | `null` | Evaluate every N steps |
| `metrics` | list[string] | `["EER"]` | Metrics to compute |

### Checkpointing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `monitor_metric` | string | `"EER"` | Metric for best model |
| `monitor_mode` | string | `"min"` | `"min"` or `"max"` |
| `save_every_epochs` | integer | `5` | Save checkpoint every N epochs |
| `early_stopping_patience` | integer | `null` | Stop if no improvement |

### Optimizer

```yaml
optimizer:
  type: "adam"
  lr: 0.0001
  weight_decay: 0.0001
  betas: [0.9, 0.999]
```

| Optimizer | Type Key | Key Args |
|-----------|----------|----------|
| Adam | `adam` | `lr`, `weight_decay`, `betas` |
| AdamW | `adamw` | `lr`, `weight_decay`, `betas` |
| SGD | `sgd` | `lr`, `momentum`, `weight_decay` |

### Scheduler

```yaml
scheduler:
  type: "cosine_annealing"
  T_max: 50
  eta_min: 0.000001
```

| Scheduler | Type Key | Key Args |
|-----------|----------|----------|
| Cosine Annealing | `cosine_annealing` | `T_max`, `eta_min` |
| Step LR | `step_lr` | `step_size`, `gamma` |
| Exponential LR | `exponential_lr` | `gamma` |

### WandB Integration

```yaml
wandb:
  enabled: true
  project: "deepfense"
  name: "experiment_1"
  entity: "your-username"  # Optional
  tags: ["wav2vec2", "aasist"]  # Optional
```

---

## Example: Minimal Configuration

```yaml
exp_name: "minimal_experiment"
output_dir: "./outputs/"
seed: 42

data:
  sampling_rate: 16000
  label_map: {bonafide: 1, spoof: 0}
  
  train:
    parquet_files: ["/path/to/train.parquet"]
    batch_size: 32
    base_transform:
      - {type: load_audio, target_sr: 16000}
      - {type: pad, max_len: 64600}
  
  val:
    parquet_files: ["/path/to/val.parquet"]
    batch_size: 64
    base_transform:
      - {type: load_audio, target_sr: 16000}
      - {type: pad, max_len: 64600}

model:
  type: "ModularDetector"
  frontend:
    type: "wav2vec2"
    args: {source: "huggingface", ckpt_path: "facebook/wav2vec2-base", freeze: true}
  backend:
    type: "MLP"
    args: {input_dim: 768, projection: [256, 64], pooling_type: "mean"}
  loss:
    type: "CrossEntropy"
    embedding_dim: 64
    n_classes: 2

training:
  epochs: 20
  device: "cuda"
  optimizer: {type: "adam", lr: 0.0001}
  scheduler: {type: "cosine_annealing", T_max: 20}
```

---

> **Next Step**: [Component Reference →](components/frontends.md)
