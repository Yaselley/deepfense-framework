# DeepFense Tutorial: Configuration Guide

This tutorial walks you through the entire DeepFense configuration file step by step. By the end, you'll understand every section and know how to customize experiments.

---

## How to Train

Everything starts with one command:

```bash
python train.py --config deepfense/config/train.yaml
```

That's it. The entire training process—model architecture, data loading, augmentations, loss functions, optimizer settings—is controlled by this single YAML file.

Let's go through each section.

---

## 1. Global Settings

```yaml
exp_name: "W2V_AASIST"
output_dir: "./outputs/"
seed: 1234
```

| Parameter | What it does |
|-----------|--------------|
| `exp_name` | Name of your experiment. A folder with this name (+ timestamp) is created in `output_dir` |
| `output_dir` | Where to save checkpoints, logs, and plots |
| `seed` | Random seed for reproducibility. Use the same seed to get identical results |

**Output structure:**
```
outputs/W2V_AASIST_20240115_143000/
├── config.yaml      # Copy of your config
├── train.log        # Training logs
├── best_model.pth   # Best checkpoint
└── ckpts/           # All checkpoints
```

---

## 2. Data Configuration

```yaml
data:
  sampling_rate: 16000
  label_map: {"bonafide": 1, "spoof": 0}
  
  train: {...}
  val: {...}
  test: {...}
```

### 2.1 Global Data Settings

| Parameter | What it does |
|-----------|--------------|
| `sampling_rate` | Target audio sample rate (Hz). Most SSL models expect 16000 |
| `label_map` | Maps string labels to integers. `bonafide: 1` means "real" = class 1 |

### 2.2 Train/Val/Test Splits

Each split follows the same structure:

```yaml
train:
  dataset_type: "StandardDataset"
  dataset_names: ["ASVSpoof19"]
  parquet_files: ["./data/train.parquet"]
  root_dir: "/path/to/audio/root"    # Optional: prepended to paths in parquet
  
  batch_size: 32
  shuffle: True
  
  base_transform: [...]
  augment_transform: [...]
```

| Parameter | What it does | Typical Values |
|-----------|--------------|----------------|
| `dataset_type` | Dataset class to use | `"StandardDataset"` |
| `parquet_files` | List of Parquet files containing metadata | Absolute or relative paths |
| `root_dir` | Base directory prepended to `path` column in parquet | Optional, for relative paths |
| `dataset_names` | Names for each parquet (for logging) | Optional |
| `batch_size` | Samples per batch | 8-64 (depends on GPU memory) |
| `shuffle` | Randomize order each epoch | `True` for train, `False` for val/test |
| `max_per_class` | Limit samples per class | Optional, for debugging |

### 2.3 Parquet Format

Your Parquet files must have these columns:

| Column | Required | Description |
|--------|----------|-------------|
| `path` | ✅ Yes | Path to audio file (absolute, or relative if using `root_dir`) |
| `label` | ✅ Yes | String label: `"bonafide"` or `"spoof"` |
| `ID` | Optional | Unique identifier (useful for leaderboard submissions) |

**Example: Creating a Parquet file**
```python
import pandas as pd

df = pd.DataFrame({
    "path": ["audio/LA_E_001.flac", "audio/LA_E_002.flac"],
    "label": ["bonafide", "spoof"],
    "ID": ["LA_E_001", "LA_E_002"]
})
df.to_parquet("train.parquet")
```

---

## 3. Transforms

Transforms process audio before it goes into the model.

### 3.1 Base Transform (Applied to all data)

```yaml
base_transform:
  - type: "pad"
    max_len: 64600      # ~4 seconds at 16kHz
    random_pad: False   # If audio > max_len: random crop (True) or take start (False)
    pad_type: "repeat"  # If audio < max_len: repeat to fill
```

| Transform | What it does |
|-----------|--------------|
| `pad` | Ensures all audio is exactly `max_len` samples |

**Calculating max_len:**
- 16000 Hz × 4 seconds = 64000 samples
- Common values: 48000 (3s), 64600 (4s), 96000 (6s)

### 3.2 Augment Transform (Training only)

```yaml
augment_transform:
  - type: "rawboost"
    noise_ratio: 0.4    # 40% chance to apply
    algo: 5             # Algorithm variant (0-8)
    
  - type: "rir"
    noise_ratio: 0.5
    csv_file: "./data/rirs.csv"
    # csv_file format: CSV with a 'path' column containing absolute paths to audio files.

```

When you list multiple augmentations, they are applied **sequentially** (one after another).

**Available Augmentations:**

| Type | What it does | Key Parameters |
|------|--------------|----------------|
| `rawboost` | Adds convolutive/impulsive noise | `algo` (0-8), `noise_ratio` |
| `rir` | Room impulse response (reverb) | `csv_file`, `noise_ratio` |
| `add_noise` | Additive background noise | `csv_file`, `snr_low`, `snr_high` |
| `add_babble` | Mix multiple speakers | `csv_file`, `speaker_count` |
| `speed_perturb` | Change speed/pitch | `speeds: [90, 100, 110]` |
| `codec` | Compression artifacts | `noise_ratio` |
| `drop_chunk` | Zero out time segments | `drop_length_low/high` |
| `drop_freq` | Apply notch filters | `drop_freq_low/high` |

**Using an Augmentation Pipeline (advanced):**

For more control, wrap augmentations in a pipeline:

```yaml
augment_transform:
  - type: "augmentation_pipeline"
    mode: "parallel"        # Pick ONE random augmentation
    concat_original: false  # Don't keep original
    p: 0.5                  # 50% chance to augment
    transforms:
      - type: "rawboost"
        noise_ratio: 1.0
      - type: "rir"
        noise_ratio: 1.0
```

| Mode | Behavior |
|------|----------|
| `parallel` | Randomly pick **one** transform |
| `sequential` | Apply **all** transforms in order |

---

## 4. Model Configuration

```yaml
model:
  type: "StandardDetector"
  
  frontend: {...}
  backend: {...}
  loss: [...]
```

The model has three components:
```
Audio → [Frontend] → Features → [Backend] → Embeddings → [Loss] → Score
```

### 4.1 Frontend (Feature Extractor)

The frontend converts raw audio into high-level features.

```yaml
frontend:
  type: "wavlm"
  args:
    source: "unil"
    ckpt_path: "./models/WavLM-Large.pt"
    freeze: True
```

**Available Frontends:**

| Type | Source | Description | Output Dim |
|------|--------|-------------|------------|
| `wav2vec2` | `fairseq` / `huggingface` | Meta's Wav2Vec 2.0 | 768 (base), 1024 (large) |
| `wavlm` | `unil` / `huggingface` | Microsoft's WavLM | 768 (base), 1024 (large) |
| `hubert` | `fairseq` / `huggingface` | Meta's HuBERT | 768 (base), 1024 (large) |
| `mert` | `huggingface` | Music foundation model | 768-1024 |
| `eat` | `huggingface` | Efficient Audio Transformer | 768 |

**Key Parameters:**

| Parameter | What it does |
|-----------|--------------|
| `source` | Where to load model from. `"huggingface"` = HuggingFace Hub, `"fairseq"` = local .pt file. DeepFense handles the different forward logics automatically. |
| `ckpt_path` | Path or HuggingFace model ID |
| `freeze` | If `True`, don't train frontend weights (recommended for fine-tuning) |

**Examples:**

```yaml
# HuggingFace WavLM
frontend:
  type: "wavlm"
  args:
    source: "huggingface"
    ckpt_path: "microsoft/wavlm-base"
    freeze: True

# Local Fairseq Wav2Vec2
frontend:
  type: "wav2vec2"
  args:
    source: "fairseq"
    ckpt_path: "/path/to/wav2vec_large.pt"
    freeze: True
```

### 4.2 Backend (Classifier)

The backend takes features and produces a fixed-size embedding.

```yaml
backend:
  type: "AASIST"
  args:
    input_dim: 1024       # Must match frontend output
    filts: [70, [1, 32], [32, 32], [32, 64], [64, 64]]
    gat_dims: [64, 32]
```

**Available Backends:**

| Type | Description | Best For |
|------|-------------|----------|
| `AASIST` | Graph Attention Network | State-of-the-art spoofing detection |
| `ECAPA_TDNN` | Speaker verification architecture | Strong generalization |
| `Nes2Net` | Res2Net-based CNN | Efficient, good performance |
| `RawNet2` | CNN + GRU | Classic architecture |
| `MLP` | Simple feedforward | Quick experiments, SSL frontends |

**Important:** `input_dim` must match your frontend's output dimension!

| Frontend | Output Dim |
|----------|------------|
| `*-base` models | 768 |
| `*-large` models | 1024 |

**MLP Backend (simplest):**

```yaml
backend:
  type: "MLP"
  args:
    input_dim: 768
    projection: [256, 64]    # Hidden layers
    pooling_type: "mean"     # mean, max, or asp (attentive)
```

### 4.3 Loss Functions

The loss module computes the training objective and produces scores for evaluation.

```yaml
loss:
  - type: "CrossEntropy"
    weight: 1.0
    embedding_dim: 64
    n_classes: 2
```

**Available Losses:**

| Type | Description | When to Use |
|------|-------------|-------------|
| `CrossEntropy` | Standard classification | Simple baseline, quick experiments |
| `OCSoftmax` | One-Class Softmax | Best for spoofing (compacts bonafide, pushes spoof) |
| `AMSoftmax` | Additive Margin Softmax | Good generalization, angular margin |
| `ASoftmax` | Angular Softmax | Alternative angular metric |

**Multi-Loss Training:**

You can combine multiple losses with weights:

```yaml
loss:
  - type: "CrossEntropy"
    weight: 0.5
    embedding_dim: 64
    n_classes: 2
    
  - type: "AMSoftmax"
    weight: 0.5
    embedding_dim: 64
    n_classes: 2
    s: 30.0
    m: 0.4
```

The total loss = 0.5 × CrossEntropy + 0.5 × AMSoftmax

**Loss Parameters:**

| Parameter | Description |
|-----------|-------------|
| `embedding_dim` | Must match backend output dimension |
| `n_classes` | Number of classes (usually 2: bonafide/spoof) |
| `weight` | Contribution to total loss |
| `s` (AMSoftmax) | Scale factor (typical: 30) |
| `m` (AMSoftmax) | Margin (typical: 0.2-0.5) |
| `w_posi`, `w_nega` (OCSoftmax) | Margins for positive/negative class |
| `alpha` (OCSoftmax) | Scale factor (typical: 20) |

**OCSoftmax (Recommended for Spoofing):**

```yaml
loss:
  - type: "OCSoftmax"
    weight: 1.0
    embedding_dim: 64
    w_posi: 0.9
    w_nega: 0.2
    alpha: 20.0
```

---

## 5. Training Configuration

```yaml
training:
  trainer: "StandardTrainer"
  epochs: 50
  device: "cuda"
  num_workers: 4
```

### 5.1 Basic Settings

| Parameter | What it does | Typical Values |
|-----------|--------------|----------------|
| `epochs` | Number of training epochs | 30-100 |
| `device` | Where to run | `"cuda"`, `"cpu"`, `"cuda:0"` |
| `num_workers` | DataLoader workers | 4-8 |

### 5.2 Logging

```yaml
batch_log_interval: 50   # Log every 50 batches
```

### 5.3 Evaluation

```yaml
eval_every_epochs: 1     # Validate every epoch
# OR
eval_every_steps: 500    # Validate every 500 steps

monitor_metric: "EER"    # Metric to track for best model
monitor_mode: "min"      # "min" = lower is better, "max" = higher is better
```

### 5.4 Optimizer

```yaml
optimizer:
  type: "adam"
  lr: 0.0001
  weight_decay: 0.0001
```

| Type | When to Use |
|------|-------------|
| `adam` | Default choice, works well for most cases |
| `adamw` | Adam with proper weight decay |
| `sgd` | Sometimes better for fine-tuning |

**Learning Rate Tips:**
- Frozen frontend: `lr: 0.0001` to `0.00001`
- End-to-end training: `lr: 0.00001` to `0.000001`
- If loss is NaN: reduce learning rate

### 5.5 Scheduler

```yaml
scheduler:
  type: "cosine_annealing"
  T_max: 50              # Should equal epochs
  eta_min: 0.000001      # Minimum LR
```

| Type | Behavior |
|------|----------|
| `cosine_annealing` | Smoothly decreases LR following cosine curve |
| `step_lr` | Drops LR by `gamma` every `step_size` epochs |
| `exponential` | Multiplies LR by `gamma` each epoch |

### 5.6 Metrics

```yaml
metrics:
  ACC: {}
  F1_SCORE: {}
  EER: {}
  minDCF:
    Pspoof: 0.05
    Cmiss: 1
    Cfa: 1
```

| Metric | Description | Good Value |
|--------|-------------|------------|
| `ACC` | Accuracy | > 95% |
| `F1_SCORE` | F1 Score | > 0.95 |
| `EER` | Equal Error Rate | < 5% |
| `minDCF` | Minimum Detection Cost | < 0.1 |

---

## 6. Complete Example Configurations

### Example 1: Quick Baseline

```yaml
exp_name: "baseline"
output_dir: "./outputs/"
seed: 42

data:
  sampling_rate: 16000
  label_map: {"bonafide": 1, "spoof": 0}
  
  train:
    parquet_files: ["./data/train.parquet"]
    root_dir: "/data/asvspoof19"
    batch_size: 32
    shuffle: True
    base_transform:
      - type: "pad"
        max_len: 64600
        
  val:
    parquet_files: ["./data/val.parquet"]
    root_dir: "/data/asvspoof19"
    batch_size: 64
    base_transform:
      - type: "pad"
        max_len: 64600

model:
  type: "StandardDetector"
  frontend:
    type: "wav2vec2"
    args:
      source: "huggingface"
      ckpt_path: "facebook/wav2vec2-base"
      freeze: True
  backend:
    type: "MLP"
    args:
      input_dim: 768
      projection: [256, 64]
      pooling_type: "mean"
  loss:
    - type: "CrossEntropy"
      embedding_dim: 64
      n_classes: 2

training:
  epochs: 20
  device: "cuda"
  optimizer:
    type: "adam"
    lr: 0.0001
  metrics:
    EER: {}
    ACC: {}
```

### Example 2: State-of-the-Art Setup

```yaml
exp_name: "wavlm_aasist_oc"
output_dir: "./outputs/"
seed: 1234

data:
  sampling_rate: 16000
  label_map: {"bonafide": 1, "spoof": 0}
  
  train:
    parquet_files: ["./data/train.parquet"]
    root_dir: "/data/asvspoof"
    batch_size: 24
    shuffle: True
    base_transform:
      - type: "pad"
        max_len: 64600
        random_pad: True
    augment_transform:
      - type: "augmentation_pipeline"
        mode: "parallel"
        p: 0.5
        transforms:
          - type: "rawboost"
            noise_ratio: 1.0
            algo: 5
          - type: "rir"
            noise_ratio: 1.0
            csv_file: "./data/rirs.csv"
          - type: "add_noise"
            noise_ratio: 1.0
            csv_file: "./data/noise.csv"
            snr_low: 5
            snr_high: 20
            
  val:
    parquet_files: ["./data/val.parquet"]
    root_dir: "/data/asvspoof"
    batch_size: 48
    base_transform:
      - type: "pad"
        max_len: 64600

model:
  type: "StandardDetector"
  frontend:
    type: "wavlm"
    args:
      source: "huggingface"
      ckpt_path: "microsoft/wavlm-large"
      freeze: True
  backend:
    type: "AASIST"
    args:
      input_dim: 1024
      filts: [70, [1, 32], [32, 32], [32, 64], [64, 64]]
      gat_dims: [64, 32]
  loss:
    - type: "OCSoftmax"
      embedding_dim: 32
      w_posi: 0.9
      w_nega: 0.2
      alpha: 20.0

training:
  epochs: 50
  device: "cuda"
  num_workers: 4
  eval_every_epochs: 1
  monitor_metric: "EER"
  monitor_mode: "min"
  optimizer:
    type: "adam"
    lr: 0.0001
    weight_decay: 0.0001
  scheduler:
    type: "cosine_annealing"
    T_max: 50
    eta_min: 0.000001
  metrics:
    EER: {}
    F1_SCORE: {}
    minDCF:
      Pspoof: 0.05
```

---

## 7. Testing Your Model

After training:

```bash
python test.py \
    --config deepfense/config/train.yaml \
    --checkpoint outputs/your_exp/best_model.pth
```

For a different test set:

```bash
python test.py \
    --config deepfense/config/train.yaml \
    --checkpoint outputs/your_exp/best_model.pth \
    --test_parquet ./data/eval.parquet
```

---

## 8. Common Issues & Solutions

| Problem | Solution |
|---------|----------|
| `CUDA out of memory` | Reduce `batch_size`, set `freeze: True` on frontend |
| `Loss is NaN` | Reduce `lr` (learning rate) |
| `EER stuck at 50%` | Model not learning - check data paths, try different lr |
| `FileNotFoundError` | Check `parquet_files` paths, use `root_dir` for relative paths |
| `KeyError: bonafide` | Check `label_map` matches your parquet's label column |

---

## Summary Cheatsheet

| Section | Key Parameters |
|---------|----------------|
| **Data** | `parquet_files`, `root_dir`, `batch_size`, `base_transform` |
| **Frontend** | `type`, `ckpt_path`, `freeze` |
| **Backend** | `type`, `input_dim` (must match frontend output) |
| **Loss** | `type`, `embedding_dim` (must match backend output), `weight` |
| **Training** | `epochs`, `lr`, `monitor_metric` |

**Quick Start Command:**
```bash
python train.py --config deepfense/config/train.yaml
```
