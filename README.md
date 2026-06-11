<div align="center">
  <img src="docs/logo/logo.png" alt="DeepFense Logo" width="400">
</div>

<div align="center">

# DeepFense

**A Modular Framework for Deepfake Audio Detection (Clip-Level & Partial Deepfake)**

[![Python](https://img.shields.io/badge/Python-3.10%2B-navy.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-navy.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-navy.svg)](LICENSE)

</div>

---

## What is DeepFense?

DeepFense lets you build deepfake audio detectors by combining **frontends** (pretrained feature extractors), **backends** (classifiers), and **loss functions** — all defined in a single YAML config. No code changes needed to run new experiments.

It supports two detection modes:

| Mode | Task | Output |
|------|------|--------|
| **Clip-level** | Full-utterance bonafide vs spoof | One score per audio clip |
| **Temporal / partial deepfake** | Per-frame bonafide vs spoof | One score per time step (e.g. 20–40 ms frames) |

```
Clip-level:     Raw Audio --> Frontend --> Backend (pools) --> Loss --> Score
Partial deepfake: Raw Audio --> Frontend --> FrameMLP --> Framewise CE --> Frame scores
```

---

## Install

```bash
conda create -n deepfense python=3.10
conda activate deepfense
pip install deepfense
```

Or install from source (for development):

```bash
conda create -n deepfense python=3.10
conda activate deepfense
cd DeepFense
pip install -e .
```

---

## Quick Start

### 1. Generate dummy test data

```bash
python tests/create_samples.py
```

### 2. Train (clip-level)

```bash
python train.py --config deepfense/config/train.yaml
```

### 2b. Train (partial deepfake / temporal)

Per-frame labels in parquet (`frame_labels` or `frame_labels_path`), full-length waveforms, and framewise metrics:

```bash
deepfense train --config deepfense/config/experiments/temporal_deepfake_example.yaml
```

See [Temporal / Partial Deepfake Guide](docs/temporal_deepfake.md) for parquet schema, label rates, and config details.

### 3. Test

```bash
python test.py \
    --config deepfense/config/train.yaml \
    --checkpoint outputs/Wav2Vec2_Nes2Net_Example_*/best_model.pth
```

### 4. Multi-GPU Training

DeepFense supports multi-GPU training out of the box via PyTorch DDP. Just use `torchrun`:

```bash
# 2 GPUs on a single node
torchrun --nproc_per_node=2 train.py --config deepfense/config/train.yaml

# 4 GPUs
torchrun --nproc_per_node=4 train.py --config deepfense/config/train.yaml
```

No config changes required -- DDP is detected automatically. Checkpoints, logs, and evaluation run on rank 0 only. The saved checkpoints are identical to single-GPU ones and can be loaded without any DDP-specific handling.

### 5. Use real data (clip-level)

Create a Parquet file with columns `ID`, `path`, `label` (`"bonafide"` / `"spoof"`), then update the config:

```yaml
data:
  train:
    parquet_files: ["/path/to/train.parquet"]
  val:
    parquet_files: ["/path/to/val.parquet"]
```

### 6. Use real data (partial deepfake)

Parquet rows need `path` plus dense frame labels (`frame_labels` list/array, or `frame_labels_path` to a `.npy` / `.npz` file). Optional clip-level `label`; if omitted, a weak label is inferred from the frames.

```yaml
data:
  sampling_rate: 16000
  label_hop_ms: 40                    # prediction rate (40 ms = 640 samples @ 16 kHz)
  # source_label_hop_ms: 20           # annotation rate if finer than label_hop_ms
  # label_merge_rule: any_spoof       # any_spoof | all_spoof | majority
  label_map: {bonafide: 1, spoof: 0}
  train:
    dataset_type: TemporalSegmentationDataset
    parquet_files: ["/path/to/train.parquet"]
  val:
    dataset_type: TemporalSegmentationDataset
    parquet_files: ["/path/to/val.parquet"]

model:
  type: TemporalDetector
  frontend:
    type: wav2vec2
    args: {source: fairseq, ckpt_path: /path/to/xlsr2_300m.pt, freeze: false}
  backend:
    type: FrameMLP
    args: {input_dim: 1024, projection: [512], activation: relu, norm_type: layer}
  loss:
    - type: FramewiseCrossEntropy
      weight: 1.0
      embedding_dim: 512
      n_classes: 2
      ignore_index: -100

training:
  monitor_metric: FRAME_F1
  monitor_mode: max
  metrics:
    FRAME_ACC: {}
    FRAME_F1: {f1_average: macro}
    FRAME_AUC: {}
    FRAME_JACCARD_SPOOF: {spoof_label: 0}
```

Partial-deepfake clips are kept at full length; variable-length batches are zero-padded in the dataloader. Trailing frame labels use `-100` and are ignored by the loss.

---

## How Configuration Works

Everything is controlled by a single YAML file. Here is the anatomy:

```yaml
# ---------- experiment ----------
exp_name: "my_experiment"
output_dir: "./outputs/"
seed: 42

# ---------- data ----------
data:
  sampling_rate: 16000
  label_map: {"bonafide": 1, "spoof": 0}
  train:
    parquet_files: ["train.parquet"]
    batch_size: 32
    base_transform:
      - type: "pad"
        max_len: 64600          # ~4 sec at 16 kHz
    augment_transform:          # training only
      - type: "rawboost"
        noise_ratio: 0.4
  val:
    parquet_files: ["val.parquet"]
    batch_size: 64
    base_transform:
      - type: "pad"
        max_len: 64600

# ---------- model ----------
model:
  type: "StandardDetector"
  frontend:
    type: "wav2vec2"                    # or wavlm, hubert, mert, eat
    args:
      source: "huggingface"             # or "fairseq" for local .pt files
      ckpt_path: "facebook/wav2vec2-xls-r-300m"
      freeze: True
  backend:
    type: "AASIST"                      # or MLP, Nes2Net, ECAPA_TDNN, RawNet2
    args:
      input_dim: 1024                   # must match frontend output dim
  loss:
    - type: "OCSoftmax"                 # or CrossEntropy, AMSoftmax, ASoftmax
      weight: 1.0
      embedding_dim: 32                 # must match backend output dim

# ---------- training ----------
training:
  epochs: 50
  device: "cuda"
  optimizer:
    type: "adam"
    lr: 0.0001
  scheduler:
    type: "cosine_annealing"
    T_max: 50
  monitor_metric: "EER"
  monitor_mode: "min"
  metrics:
    EER: {}
    ACC: {}
    minDCF: {Pspoof: 0.05}
```

See the [Full Tutorial](docs/03_full_tutorial.md) for a detailed walkthrough of every parameter.

---

## Available Components

| Category | Options |
|----------|---------|
| **Detectors** | `StandardDetector` (clip-level), `TemporalDetector` (partial deepfake) |
| **Datasets** | `StandardDataset`, `TemporalSegmentationDataset` |
| **Frontends** | Wav2Vec2, WavLM, HuBERT, MERT, EAT |
| **Backends** | AASIST, ECAPA-TDNN, Nes2Net, RawNet2, MLP, TCM, **FrameMLP** |
| **Losses** | CrossEntropy, OC-Softmax, AM-Softmax, A-Softmax, **FramewiseCrossEntropy** |
| **Augmentations** | RawBoost, RIR, Codec, AdditiveNoise, SpeedPerturb, AddBabble, DropChunk, DropFreq |
| **Metrics (clip)** | EER, minDCF, actDCF, ACC, F1 |
| **Metrics (temporal)** | FRAME_ACC, FRAME_F1, FRAME_AUC, FRAME_JACCARD_SPOOF |

List them from the CLI:

```bash
deepfense list
deepfense list --component-type backends
```

---

## Pretrained Models & Datasets (HuggingFace Hub)

DeepFense publishes **455+ pretrained models** and **12 datasets** at [huggingface.co/DeepFense](https://huggingface.co/DeepFense).

```bash
# See what's available
deepfense download list-datasets
deepfense download list-models --filter WavLM

# Download a dataset (parquet files)
deepfense download dataset CompSpoof

# Download a pretrained model (checkpoint + config)
deepfense download model ASV19_WavLM_Nes2Net_NoAug_Seed42

# Test the downloaded model
python test.py \
    --config models/ASV19_WavLM_Nes2Net_NoAug_Seed42/config.yaml \
    --checkpoint models/ASV19_WavLM_Nes2Net_NoAug_Seed42/best_model.pth
```

Or in Python:

```python
from deepfense.hub import download_dataset, download_model

parquets = download_dataset("CompSpoof")           # returns list of local paths
files    = download_model("ASV19_WavLM_Nes2Net_NoAug_Seed42")  # returns {"checkpoint": ..., "config": ...}
```

See the [HuggingFace Hub Guide](docs/07_huggingface_hub.md) for full workflows (training, evaluation, inference).

---

## Adding Your Own Component

Every component type follows the same pattern:

1. Create a file (e.g. `deepfense/models/backends/my_backend.py`)
2. Decorate with the registry:
   ```python
   from deepfense.utils.registry import register_backend
   from deepfense.models.base_model import BaseBackend

   @register_backend("MyBackend")
   class MyBackend(BaseBackend):
       def __init__(self, config):
           super().__init__()
           # ...

       def forward(self, x):
           # ...
   ```
3. Import it in the package `__init__.py`
4. Use it in your config:
   ```yaml
   backend:
     type: "MyBackend"
     args: { ... }
   ```

The same pattern applies to frontends, losses, augmentations, datasets, optimizers, and metrics. See the [user guides](docs/user_guide/) for detailed walkthroughs.

---

## Project Structure

```
DeepFense/
├── train.py, test.py          # Entry points
├── deepfense/
│   ├── cli/                   # CLI commands (train, test, list, download)
│   ├── config/                # YAML configs + parquet generators
│   │   └── experiments/       # temporal_deepfake_example.yaml, batch generators
│   ├── data/
│   │   ├── detection_dataset.py      # StandardDataset (clip-level)
│   │   ├── temporal_dataset.py       # TemporalSegmentationDataset (partial deepfake)
│   │   ├── temporal_utils.py         # Frame label alignment / downsampling
│   │   └── transforms/               # Augmentations, RawBoost, audio utils
│   ├── models/
│   │   ├── detector.py               # StandardDetector
│   │   ├── temporal_detector.py      # TemporalDetector
│   │   ├── frontends/                # Wav2Vec2, WavLM, HuBERT, MERT, EAT
│   │   ├── backends/                 # AASIST, MLP, FrameMLP, ...
│   │   ├── losses/                   # CrossEntropy, FramewiseCrossEntropy, ...
│   │   └── modules/                  # Shared layers (pooling, conformer, fairseq_local)
│   ├── training/              # Trainer, evaluator, metrics, seed
│   └── utils/                 # Registry, visualization, optional trace helpers
├── docs/
│   └── temporal_deepfake.md   # Partial deepfake design & config reference
├── wedefense/                 # Bundled PartialSpoof / localization tooling (legacy)
└── scripts/                   # Data prep, prediction export, analysis
```

---

## Documentation

| Guide | Description |
|-------|-------------|
| [Installation](docs/01_installation.md) | Setup instructions |
| [Quick Start](docs/02_quickstart.md) | First model in 5 minutes |
| [Full Tutorial](docs/03_full_tutorial.md) | Every config option explained |
| [Architecture](docs/04_architecture.md) | How DeepFense works internally |
| [Configuration Reference](docs/05_configuration.md) | All YAML parameters |
| [Library Usage](docs/06_library_usage.md) | Use DeepFense as a Python library |
| [HuggingFace Hub](docs/07_huggingface_hub.md) | Download datasets & pretrained models |
| [**Temporal / Partial Deepfake**](docs/temporal_deepfake.md) | Per-frame labels, `TemporalDetector`, framewise metrics |
| [CLI Reference](docs/cli_reference.md) | CLI commands |
| [Components](docs/components/) | Frontend, backend, loss, augmentation reference |
| [User Guides](docs/user_guide/) | Adding custom components, training workflows |

For PartialSpoof localization metrics, RTTM export, and legacy recipes, see the bundled `wedefense/` tree and `wedefense/README.md`.

---

## Citation

```bibtex
@software{deepfense2025,
  title={DeepFense: A Modular Framework for Deepfake Audio Detection},
  author={DeepFense Team},
  year={2025},
  url={https://github.com/Yaselley/deepfense-framework}
}
```

## License

Apache 2.0 -- see [LICENSE](LICENSE) for details.
