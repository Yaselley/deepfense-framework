# Getting Started: Training & Configuration

This tutorial walks you through training your first model using DeepFense and explains every configuration parameter in detail.

## 🚀 Running a Training Experiment

The primary entry point is `train.py`. You run it by providing a configuration file.

```bash
python train.py --config deepfense/config/train.yaml
```

### Resume Training
To resume from a checkpoint (e.g., if the run crashed):
```bash
python train.py --config deepfense/config/train.yaml --resume outputs/ExpName/ckpts/ckpt_epoch005_step001000.pth
```

---

## ⚙️ Configuration Manual (`train.yaml`)

The configuration file is split into four main sections: **Data**, **Model**, **Training**, and **Top-Level**.

### 1. Top-Level Settings

```yaml
exp_name: "Baseline_Model"   # Name used for the output folder
output_dir: "./outputs/"     # Where output folders are created
seed: 42                     # Random seed for reproducibility
```

### 2. Data Configuration (`data`)

Controls how data is loaded and augmented.

```yaml
data:
  sampling_rate: 16000       # Target SR. All audio is resampled to this.
  
  # Label Mapping: Defines which integer represents which class
  label_map:
    bonafide: 1
    spoof: 0

  train:
    dataset_type: "StandardDataset"  # Class name registered in deepfense/data/detection_dataset.py
    
    # List of Parquet files to load for training
    parquet_files:
      - "/path/to/train.parquet"
    
    # Optional: Names for these datasets (for logging)
    dataset_names:
      - "ASVSpoof19_Train"

    batch_size: 32
    shuffle: true
    num_workers: 4       # Number of CPU threads for loading data
    
    # 🔄 Augmentation Pipeline
    augment_transform:
      - type: "augmentation_pipeline"
        mode: "sequential" # Options: "sequential", "parallel", "concat"
        transforms:
          - type: "rawboost"
            algo: [0, 1, 2]      # Algo types for RawBoost
            noise_ratio: 1.0     # Probability of applying
          - type: "speed_perturb"
            speeds: [95, 105]    # Speed % (slower/faster)

  val:
    # Same structure as train, usually no augmentations
    dataset_type: "StandardDataset"
    parquet_files:
      - "/path/to/dev.parquet"
    batch_size: 16
    shuffle: false
```

### 3. Model Configuration (`model`)

Defines the neural network architecture.

```yaml
model:
  type: "StandardDetector"
  
  # 👂 Frontend: Audio -> Features
  frontend:
    type: "wav2vec2"       # Options: wav2vec2, wavlm, hubert, mert, eat
    args:
      source: "fairseq"    # "fairseq", "unil", or "huggingface"
      ckpt_path: "/path/to/xlsr_53.pt" # Or HF ID like "m-a-p/MERT-v1-95M"
      freeze: true         # If true, frontend weights are not updated

  # 🧠 Backend: Features -> Embedding
  backend:
    type: "MLP"            # Options: MLP, AASIST, Res2Net, TCM
    args:
      input_dim: 1024      # Must match Frontend output dimension
      projection: [128]    # Hidden layers
      pooling_type: "asp"  # Attentive Statistics Pooling

  # 📉 Loss: Embedding -> Score
  loss:
    - type: "AMSoftmax"    # Options: CrossEntropy, AMSoftmax, OCSoftmax
      weight: 1.0          # Weight in total loss calculation
      embedding_dim: 128   # Must match Backend output dimension
      n_classes: 2
      m: 0.3               # Margin parameter
      s: 30                # Scale parameter
```

### 4. Training Configuration (`training`)

Controls the optimization process.

```yaml
training:
  trainer: "StandardTrainer"
  epochs: 20
  gradient_accumulation_steps: 1 # Simulate larger batches
  
  # Optimizer
  optimizer:
    type: "adam"           # Options: adam, sgd, adamw
    lr: 0.0001
    weight_decay: 0.0001
    
  # Scheduler (Optional)
  scheduler:
    type: "cosine"
    T_max: 20

  # 📊 Evaluation Metrics
  metrics:
    EER: {}                # Equal Error Rate
    minDCF:                # Minimum Detection Cost Function
      Pspoof: 0.05
      Cmiss: 1
      Cfa: 1

  # Logging & Checkpointing
  batch_log_interval: 10   # Log loss every N steps
  eval_every_epochs: 1     # Run validation every N epochs
  device: "cuda"           # cuda or cpu
```
