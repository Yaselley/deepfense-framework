# DeepFense Complete Pipeline Flow

This document provides a comprehensive overview of the complete pipeline from data preparation to model deployment.

---

## Overview

DeepFense follows a modular pipeline architecture where data flows through distinct stages:

```
Data → Preprocessing → Augmentation → Frontend → Backend → Loss → Optimization → Evaluation
```

---

## Stage 1: Data Preparation

### 1.1 Protocol Files to Parquet

**Input**: Protocol files (e.g., ASVspoof format)
```
LA_0079 LA_T_1138215 - - bonafide
LA_0080 LA_T_1138216 - - spoof
```

**Process**: Run data generation script
```bash
deepfense generate-data deepfense/config/parquets/generate_asv19.py
```

**Output**: Parquet files with metadata
```python
# train.parquet
ID          path                          label      dataset_name
LA_T_1138215 /path/to/LA_T_1138215.flac  bonafide   ASVSpoof19
LA_T_1138216 /path/to/LA_T_1138216.flac  spoof       ASVSpoof19
```

### 1.2 Parquet File Structure

Required columns:
- `ID`: Unique identifier (string)
- `path`: Path to audio file (string, absolute or relative)
- `label`: Label string ("bonafide" or "spoof")
- `dataset_name`: (Optional) Dataset identifier

Optional columns:
- `speaker_id`: Speaker identifier
- `recording_device`: Device used
- `environment`: Recording environment

---

## Stage 2: Data Loading

### 2.1 Dataset Initialization

**Component**: `DetectionDataset` or custom dataset

**Process**:
1. Read parquet file(s)
2. Map labels to integers using `label_map`
3. Store metadata in memory

**Config**:
```yaml
data:
  train:
    dataset_type: "DetectionDataset"
    parquet_files: ["/path/to/train.parquet"]
    label_map:
      bonafide: 1
      spoof: 0
```

### 2.2 Audio Loading

**Process** (in `__getitem__`):
1. Read audio file using `soundfile` or `librosa`
2. Resample to target sample rate (if needed)
3. Convert to mono (if `mono: True`)
4. Return as numpy array

**Output**: Raw audio waveform `[Time]` or `[Channels, Time]`

---

## Stage 3: Data Preprocessing

### 3.1 Base Transforms

**Applied**: Always (both training and validation)

**Common Transforms**:
- **Padding**: `PadToLength` - Pad or crop to fixed length
- **Resampling**: Automatic if `target_sr` differs from file SR
- **Mono Conversion**: If `mono: True`

**Config**:
```yaml
data:
  train:
    base_transform:
      - type: "PadToLength"
        args: {length: 160000}  # 10 seconds at 16kHz
```

**Output**: Preprocessed audio `[Time]`

### 3.2 Augmentation Transforms

**Applied**: Only during training (probabilistic)

**Common Augmentations**:
- **RawBoost**: Advanced augmentation suite (noise, filtering, etc.)
- **RIR**: Room Impulse Response simulation
- **Codec**: Audio codec compression simulation
- **AdditiveNoise**: Gaussian noise addition
- **RandomCrop**: Random cropping/padding
- **SpeedPerturb**: Speed variation

**Config**:
```yaml
data:
  train:
    augment_transform:
      - type: "RawBoost"
        args: {noise_ratio: 0.5, algo: 5}
      - type: "RIR"
        args: {noise_ratio: 0.3, csv_file: "path/to/rir.csv"}
      - type: "AdditiveNoise"
        args: {noise_ratio: 0.2, snr_range: [5, 15]}
```

**Output**: Augmented audio `[Time]`

---

## Stage 4: Model Forward Pass

### 4.1 Frontend (Feature Extraction)

**Input**: Raw audio `[Batch, Time]`
- Example: `[32, 160000]` = 32 samples, 10 seconds each at 16kHz

**Process**:
1. Load pretrained model (Wav2Vec2, WavLM, etc.)
2. Extract features from audio
3. Return feature representations

**Output**: Features `[Batch, Time', Dim]`
- Example: `[32, 500, 768]` = 32 samples, 500 time steps, 768 dimensions

**Available Frontends**:
- Wav2Vec2 (768 dim)
- WavLM (768 dim)
- HuBERT (768 dim)
- EAT (768 dim)
- MERT (768 dim)

**Config**:
```yaml
model:
  frontend:
    type: "wav2vec2"
    args:
      ckpt_path: "/path/to/wav2vec2.pt"
      freeze: True
      output_dim: 768
```

### 4.2 Backend (Classification)

**Input**: Features `[Batch, Time', Dim]`
- Example: `[32, 500, 768]`

**Process**:
1. Pool over time dimension (mean, attention, etc.)
2. Process through classification network
3. Return fixed-size embeddings

**Output**: Embeddings `[Batch, EmbeddingDim]`
- Example: `[32, 128]` = 32 samples, 128-dimensional embeddings

**Available Backends**:
- AASIST (GAT-based)
- ECAPA-TDNN (Channel attention)
- RawNet2 (CNN-GRU)
- MLP (Simple MLP)
- Nes2Net (Res2Net-based)
- TCM (Conformer-based)

**Config**:
```yaml
model:
  backend:
    type: "AASIST"
    args:
      input_dim: 768  # Must match frontend output
      output_dim: 128
```

### 4.3 Loss Function (Scoring)

**Input**: 
- Embeddings `[Batch, EmbeddingDim]`
- Labels `[Batch]`

**Process**:
1. Project embeddings to logits/scores
2. Compute loss
3. Return loss value and scores

**Output**:
- Loss: Scalar tensor
- Scores: `[Batch]` or `[Batch, n_classes]`

**Available Losses**:
- CrossEntropy
- OC-Softmax
- AM-Softmax
- A-Softmax

**Config**:
```yaml
model:
  loss:
    - type: "CrossEntropy"
      args:
        embedding_dim: 128
        n_classes: 2
```

---

## Stage 5: Training Loop

### 5.1 Training Step

**Process**:
1. Load batch of data
2. Apply transforms and augmentations
3. Forward pass through model
4. Compute loss
5. Backward pass
6. Optimizer step
7. Update learning rate (if scheduler)

**Config**:
```yaml
training:
  optimizer:
    type: "Adam"
    args:
      lr: 0.0001
  scheduler:
    type: "CosineAnnealingLR"
    args:
      T_max: 100
  epochs: 100
```

### 5.2 Validation Step

**Process** (every N epochs):
1. Set model to eval mode
2. Run forward pass on validation set (no augmentations)
3. Collect scores and labels
4. Compute metrics (EER, minDCF, F1, ACC)
5. Save best model if metric improves

**Metrics Computed**:
- EER (Equal Error Rate)
- minDCF (minimum Detection Cost Function)
- F1_SCORE
- ACC (Accuracy)
- actDCF (Actual Detection Cost Function)

---

## Stage 6: Testing/Evaluation

### 6.1 Test Process

**Input**: Trained model checkpoint

**Process**:
1. Load model from checkpoint
2. Run inference on test set
3. Collect scores and labels
4. Compute metrics per dataset
5. Save predictions to files

**Output**:
- `results.json`: Overall and per-dataset metrics
- `predictions/*.txt`: Per-dataset prediction files

### 6.2 Prediction Format

Each prediction file contains:
```
ID_audio,label,score_class0,score_class1
LA_T_1138215,1,0.1234,0.8766
LA_T_1138216,0,0.9123,0.0877
```

---

## Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DeepFense Complete Pipeline                        │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────┐
│ Protocol     │
│ Files        │
└──────┬───────┘
       │
       ▼
┌──────────────┐      deepfense generate-data
│ Parquet      │ ◄──────────────────────────────
│ Files        │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Dataset      │  Read metadata, map labels
│ Loading      │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Audio        │  Load audio files
│ Loading      │  (soundfile/librosa)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Base         │  Padding, resampling, mono
│ Transforms   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Augmentation │  RawBoost, RIR, Noise
│ (Training)   │  (probabilistic)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Frontend     │  Wav2Vec2/WavLM/HuBERT
│ Feature      │  Audio → Features
│ Extraction   │  [B, T] → [B, T', D]
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Backend      │  AASIST/MLP/ECAPA
│ Classification│ Features → Embeddings
│              │  [B, T', D] → [B, E]
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Loss         │  CrossEntropy/OC-Softmax
│ Computation  │  Embeddings → Loss + Scores
│              │  [B, E] → Loss + [B]
└──────┬───────┘
       │
       ├─────────────────┐
       │                 │
       ▼                 ▼
┌──────────────┐  ┌──────────────┐
│ Optimization │  │ Evaluation   │
│ Backward     │  │ Metrics      │
│ Optimizer    │  │ (EER, F1)    │
│ Scheduler    │  │              │
└──────────────┘  └──────────────┘
       │                 │
       ▼                 ▼
┌──────────────┐  ┌──────────────┐
│ Checkpoint   │  │ Results       │
│ Saving       │  │ JSON + TXT    │
└──────────────┘  └──────────────┘
```

---

## Data Shapes Throughout Pipeline

| Stage | Shape | Example | Description |
|-------|-------|---------|-------------|
| **Raw Audio** | `[Time]` | `[160000]` | Single audio file, 10 seconds at 16kHz |
| **Batch Audio** | `[Batch, Time]` | `[32, 160000]` | Batch of 32 samples |
| **Frontend Output** | `[Batch, Time', Dim]` | `[32, 500, 768]` | Features with reduced time dimension |
| **Backend Output** | `[Batch, EmbeddingDim]` | `[32, 128]` | Fixed-size embeddings |
| **Loss Scores** | `[Batch]` or `[Batch, Classes]` | `[32]` or `[32, 2]` | Classification scores |
| **Labels** | `[Batch]` | `[32]` | Ground truth labels |

---

## Configuration Flow

The configuration file drives the entire pipeline:

```yaml
# 1. Data Configuration
data:
  train:
    parquet_files: [...]      # → Dataset loading
    base_transform: [...]     # → Preprocessing
    augment_transform: [...]   # → Augmentation

# 2. Model Configuration
model:
  frontend: {...}             # → Feature extraction
  backend: {...}               # → Classification
  loss: [...]                  # → Loss computation

# 3. Training Configuration
training:
  optimizer: {...}            # → Optimization
  scheduler: {...}            # → LR scheduling
  metrics: {...}              # → Evaluation
```

---

## Key Design Principles

1. **Modularity**: Each stage is independent and swappable
2. **Configuration-Driven**: Everything defined in YAML
3. **Registry Pattern**: Components registered via decorators
4. **Reproducibility**: Configs saved with every experiment
5. **Extensibility**: Easy to add new components

---

## Next Steps

- See [Architecture Overview](04_architecture.md) for detailed component design
- See [Configuration Reference](05_configuration.md) for all parameters
- See [Training Guide](user_guide/training_with_cli.md) for training workflows
- See [Extending DeepFense](user_guide/extending.md) for adding components

