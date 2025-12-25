# Quick Start: Train in 5 Minutes

This guide gets you from zero to a trained model as fast as possible.

---

## Prerequisites

- DeepFense installed ([Installation Guide](01_installation.md))
- A dataset in Parquet format (or we'll create a dummy one)

---

## Step 1: Create Sample Data

If you don't have a dataset yet, create a dummy Parquet file for testing:

```python
import pandas as pd

# Create dummy training data
train_data = pd.DataFrame({
    "path": [
        "/path/to/audio1.flac",
        "/path/to/audio2.flac",
        "/path/to/audio3.flac",
        "/path/to/audio4.flac",
    ],
    "label": ["bonafide", "spoof", "bonafide", "spoof"],
    "ID": ["train_001", "train_002", "train_003", "train_004"]
})
train_data.to_parquet("train.parquet")

# Create dummy validation data
val_data = pd.DataFrame({
    "path": [
        "/path/to/val_audio1.flac",
        "/path/to/val_audio2.flac",
    ],
    "label": ["bonafide", "spoof"],
    "ID": ["val_001", "val_002"]
})
val_data.to_parquet("val.parquet")

print("Created train.parquet and val.parquet")
```

> **Important**: Replace `/path/to/audioX.flac` with actual paths to your audio files!

---

## Step 2: Update Config

Edit `deepfense/config/train.yaml` to point to your data:

```yaml
data:
  train:
    parquet_files: ["/absolute/path/to/train.parquet"]
  val:
    parquet_files: ["/absolute/path/to/val.parquet"]
```

---

## Step 3: Run Training

```bash
python train.py --config deepfense/config/train.yaml
```

You'll see output like:
```
[2024-01-15 10:30:00] [INFO] [train] Experiment directory: outputs/default_exp_20240115_103000
[2024-01-15 10:30:05] [INFO] [trainer] Trainable parameters: 1,234,567
Epoch 1/50: 100%|██████████| 125/125 [02:30<00:00]
[2024-01-15 10:32:35] [INFO] [trainer] --- End-of-Epoch Validation (Epoch 1) ---
[2024-01-15 10:32:35] [INFO] [trainer] 📈 Average Metrics: loss: 0.4532, EER: 15.23
```

---

## Step 4: Test Your Model

After training completes:

```bash
python test.py \
    --config deepfense/config/train.yaml \
    --checkpoint outputs/default_exp_20240115_103000/best_model.pth
```

---

## Output Structure

After training, check `outputs/YOUR_EXP_NAME/`:

```
outputs/default_exp_20240115_103000/
├── config.yaml          # Saved configuration
├── train.log            # Full training log
├── best_model.pth       # Best checkpoint
├── ckpts/               # All checkpoints
│   ├── ckpt_epoch001_step000125.pth
│   └── ...
├── results/             # Metrics JSON files
│   └── metrics_epoch1_step125.json
└── plots/               # Training curves
    ├── trend_loss.png
    └── trend_EER.png
```

---

## What's Next?

- 📚 **[Full Tutorial](03_full_tutorial.md)** — Complete walkthrough with real data
- 🏗️ **[Architecture](04_architecture.md)** — Understand how DeepFense works
- ⚙️ **[Configuration](05_configuration.md)** — Customize your experiments

---

> **Need help?** Check the [Troubleshooting](#troubleshooting) section in the Installation guide.
