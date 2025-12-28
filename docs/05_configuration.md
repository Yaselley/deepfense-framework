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

**Parameters:**

* **exp_name** - (*str*) Name for the experiment folder (default: `"default_exp"`).
* **output_dir** - (*str*) Base directory for outputs (default: `"./outputs/"`).
* **seed** - (*int*) Random seed for reproducibility (default: `42`).

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

**Parameters:**

* **dataset_type** - (*str*) Dataset class name (default: `"StandardDataset"`).
* **parquet_files** - (*list[str]*) **Required**. Paths to Parquet files.
* **dataset_names** - (*list[str]*) Names for each Parquet file (default: `null`).
* **batch_size** - (*int*) Batch size (default: `32`).
* **shuffle** - (*bool*) Shuffle data (default: `true`).
* **num_workers** - (*int*) DataLoader workers (default: `4`).
* **drop_last** - (*bool*) Drop incomplete batches (default: `false`).

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

**Available Transforms:**

* **load_audio** - Parameters: `target_sr`, `mono`. Loads audio file.
* **pad** - Parameters: `max_len`, `random_pad`, `pad_type`. Pads/truncates to fixed length.

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

**Modes:**

* **parallel / chain** - Picks ONE transform randomly.
* **sequential / chain** - Applies ALL transforms in order.
* **sequential / independent** - Creates separate copies with each transform.

#### Available Augmentations

Complete details in [Augmentation Reference](components/augmentations.md).

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

### Frontend

**Supported Frontends:**

* **Wav2Vec2** (`wav2vec2`) - Parameters: `source`, `ckpt_path`, `freeze`.
* **WavLM** (`wavlm`) - Parameters: `source`, `ckpt_path`, `freeze`.
* **HuBERT** (`hubert`) - Parameters: `source`, `ckpt_path`, `freeze`.
* **MERT** (`mert`) - Parameters: `ckpt_path`, `freeze`, `trust_remote_code`.
* **EAT** (`eat`) - Parameters: `ckpt_path`, `freeze`, `trust_remote_code`.

Detailed usage in [Frontend Reference](components/frontends.md).

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

### Backend

**Supported Backends:**

* **AASIST** (`AASIST`) - Parameters: `input_dim`, `filts`, `gat_dims`.
* **ECAPA-TDNN** (`ECAPA_TDNN`) - Parameters: `channels`, `emb_dim`.
* **RawNet2** (`RawNet2`) - Parameters: `filts`, `gru_node`, `emb_dim`.
* **MLP** (`MLP`) - Parameters: `input_dim`, `projection`, `pooling_type`.
* **Res2Net** (`Nes2Net`) - Parameters: `strides`, `filts`.

Detailed usage in [Backend Reference](components/backends.md).

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

### Loss Functions

**Supported Losses:**

* **OC-Softmax** (`OCSoftmax`) - Parameters: `embedding_dim`, `m_real`, `m_fake`, `alpha`.
* **AM-Softmax** (`AMSoftmax`) - Parameters: `embedding_dim`, `n_classes`, `m`, `s`.
* **A-Softmax** (`ASoftmax`) - Parameters: `embedding_dim`, `n_classes`, `m`.
* **Cross Entropy** (`CrossEntropy`) - Parameters: `embedding_dim`, `n_classes`.

Detailed usage in [Loss Reference](components/losses.md).

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

### Core Parameters

**Parameters:**

* **trainer** - (*str*) Trainer class (default: `"StandardTrainer"`).
* **device** - (*str*) Device (`cuda`, `cpu`, `cuda:0`) (default: `"cuda"`).
* **epochs** - (*int*) Total training epochs (default: `50`).
* **gradient_accumulation_steps** - (*int*) Accumulate gradients (default: `1`).
* **max_grad_norm** - (*float*) Gradient clipping norm (default: `1.0`).

### Logging

### Logging

**Parameters:**

* **batch_log_interval** - (*int*) Log every N batches (default: `50`).

### Evaluation

### Evaluation

**Parameters:**

* **eval_every_epochs** - (*int*) Evaluate every N epochs (default: `1`).
* **eval_every_steps** - (*int*) Evaluate every N steps (default: `null`).
* **metrics** - (*list[str]*) Metrics to compute (default: `["EER"]`).

### Checkpointing

### Checkpointing

**Parameters:**

* **monitor_metric** - (*str*) Metric for best model (default: `"EER"`).
* **monitor_mode** - (*str*) `"min"` or `"max"` (default: `"min"`).
* **save_every_epochs** - (*int*) Save checkpoint every N epochs (default: `5`).
* **early_stopping_patience** - (*int*) Stop if no improvement (default: `null`).

### Optimizer

```yaml
optimizer:
  type: "adam"
  lr: 0.0001
  weight_decay: 0.0001
  betas: [0.9, 0.999]
```

### Optimizer

Detailed reference in [Optimizers & Schedulers](components/optimizers_schedulers.md).

**Common Parameters:**

* **lr** - (*float*) Learning rate (default: `1e-6`).
* **weight_decay** - (*float*) L2 penalty (default: `1e-4`).
* **betas** - (*tuple*) (0.9, 0.999).

### Scheduler

```yaml
scheduler:
  type: "cosine_annealing"
  T_max: 50
  eta_min: 0.000001
```

### Scheduler

Detailed reference in [Optimizers & Schedulers](components/optimizers_schedulers.md).

**Common Parameters:**

* **T_max** - (*int*) Max iterations.
* **gamma** - (*float*) Decay factor.
* **eta_min** - (*float*) Min learning rate.

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
