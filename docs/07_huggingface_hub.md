# Using HuggingFace Hub

DeepFense publishes **datasets** and **pretrained models** on HuggingFace at [huggingface.co/DeepFense](https://huggingface.co/DeepFense).

This guide shows how to use them.

---

## Available Datasets

| Dataset | Samples | Description |
|---------|---------|-------------|
| [ASVSpoof19](https://huggingface.co/datasets/DeepFense/ASVSpoof19) | -- | ASVspoof 2019 LA |
| [CompSpoof](https://huggingface.co/datasets/DeepFense/CompSpoof) | 2.5k | Mixed bonafide/spoof compositions |
| [CtrSVDD](https://huggingface.co/datasets/DeepFense/CtrSVDD) | 93k | Singing voice deepfake detection |
| [DECRO](https://huggingface.co/datasets/DeepFense/DECRO) | 118k | Cross-domain deepfake detection |
| [EnvSDD](https://huggingface.co/datasets/DeepFense/EnvSDD) | 40k | Environmental sound deepfake detection |
| [FakeMusicCaps](https://huggingface.co/datasets/DeepFense/FakeMusicCaps) | 28k | Fake music generation detection |
| [ODSS](https://huggingface.co/datasets/DeepFense/ODSS) | 30k | Open-domain speech spoofing |
| [SONICS](https://huggingface.co/datasets/DeepFense/SONICS) | 49k | Spoofing in noisy conditions |
| [SpeechFake](https://huggingface.co/datasets/DeepFense/SpeechFake) | 4.5M | Large-scale speech deepfake |
| [SpoofCeleb](https://huggingface.co/datasets/DeepFense/SpoofCeleb) | 2.6M | Celebrity voice spoofing |
| [VCapAV](https://huggingface.co/datasets/DeepFense/VCapAV) | 90k | Audio-visual deepfake |
| [WaveFake](https://huggingface.co/datasets/DeepFense/WaveFake) | 134k | Waveform-based fake detection |

Each dataset contains parquet files (`*_train.parquet`, `*_dev.parquet`, `*_eval.parquet`) with columns: `ID`, `path`, `label`, `dataset_name`.

---

## Available Models (455+)

Models follow the naming convention: `{Dataset}_{Frontend}_{Backend}_{Aug}_{Seed}`

Examples:
- `ASV19_WavLM_Nes2Net_NoAug_Seed42`
- `CodecFake_EAT_Nes2Net_NoAug_Seed240`
- `FakeMusicCaps_mert_AASIST_NoAug_Seed42`
- `HABLA_EAT_MLP_NoAug_Seed2`

Each model repo contains:
- `best_model.pth` -- the trained checkpoint
- `config.yaml` -- the exact config used for training

---

## Download via CLI

### List what's available

```bash
# List all datasets
deepfense download list-datasets

# List all models (shows first 20)
deepfense download list-models

# Filter models
deepfense download list-models --filter WavLM
deepfense download list-models --filter ASV19 --limit 50
```

### Download a dataset

```bash
deepfense download dataset CompSpoof
# Downloads to ./data/CompSpoof/
#   CompSpoof_train.parquet
#   CompSpoof_dev.parquet
#   CompSpoof_eval.parquet

deepfense download dataset ASVSpoof19 --output-dir ./my_data
```

### Download a model

```bash
deepfense download model ASV19_WavLM_Nes2Net_NoAug_Seed42
# Downloads to ./models/ASV19_WavLM_Nes2Net_NoAug_Seed42/
#   best_model.pth
#   config.yaml
```

---

## Download via Python

```python
from deepfense.hub import download_dataset, download_model, list_models

# Download a dataset
parquet_files = download_dataset("CompSpoof", output_dir="./data")
# Returns: ["/abs/path/data/CompSpoof/CompSpoof_train.parquet", ...]

# Download a model
model_files = download_model("ASV19_WavLM_Nes2Net_NoAug_Seed42")
# Returns: {"checkpoint": "/abs/path/...", "config": "/abs/path/..."}

# Search for models
models = list_models(pattern="WavLM")
```

---

## Complete Workflows

### Workflow 1: Train on a HuggingFace dataset

```bash
# 1. Download the dataset
deepfense download dataset CompSpoof

# 2. The audio files still need to be on disk.
#    The parquet files contain relative paths.
#    Download the actual audio following each dataset's README.

# 3. Create a config pointing to the downloaded parquets
```

```yaml
# config_compspoof.yaml
exp_name: "CompSpoof_WavLM_AASIST"
output_dir: "./outputs/"
seed: 42

data:
  sampling_rate: 16000
  label_map: {"bonafide": 1, "spoof": 0}
  train:
    dataset_type: "StandardDataset"
    dataset_names: ["CompSpoof"]
    parquet_files: ["./data/CompSpoof/CompSpoof_train.parquet"]
    root_dir: "/path/to/compspoof/audio"  # where the audio files live
    batch_size: 24
    shuffle: True
    base_transform:
      - type: "pad"
        max_len: 64600
  val:
    dataset_type: "StandardDataset"
    dataset_names: ["CompSpoof"]
    parquet_files: ["./data/CompSpoof/CompSpoof_dev.parquet"]
    root_dir: "/path/to/compspoof/audio"
    batch_size: 48
    base_transform:
      - type: "pad"
        max_len: 64600
  test:
    dataset_type: "StandardDataset"
    dataset_names: ["CompSpoof"]
    parquet_files: ["./data/CompSpoof/CompSpoof_eval.parquet"]
    root_dir: "/path/to/compspoof/audio"
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
  loss:
    - type: "OCSoftmax"
      embedding_dim: 32
      w_posi: 0.9
      w_nega: 0.2
      alpha: 20.0

training:
  epochs: 50
  device: "cuda"
  optimizer:
    type: "adam"
    lr: 0.0001
  monitor_metric: "EER"
  monitor_mode: "min"
  metrics:
    EER: {}
    ACC: {}
```

```bash
# 4. Train
python train.py --config config_compspoof.yaml
```

### Workflow 2: Evaluate a pretrained model

```bash
# 1. Download the model
deepfense download model ASV19_WavLM_Nes2Net_NoAug_Seed42

# 2. Look at the config it was trained with
cat models/ASV19_WavLM_Nes2Net_NoAug_Seed42/config.yaml

# 3. Test it (update the data.test section to point to your test data)
python test.py \
    --config models/ASV19_WavLM_Nes2Net_NoAug_Seed42/config.yaml \
    --checkpoint models/ASV19_WavLM_Nes2Net_NoAug_Seed42/best_model.pth
```

### Workflow 3: Inference on a single file

```python
import torch
import torchaudio
from omegaconf import OmegaConf
from deepfense.hub import download_model
from deepfense.utils.registry import build_detector
from deepfense.models import *

# 1. Download
files = download_model("ASV19_WavLM_Nes2Net_NoAug_Seed42")

# 2. Load config and build model
cfg = OmegaConf.load(files["config"])
model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
model = build_detector(cfg.model.type, model_cfg)

# 3. Load checkpoint
state = torch.load(files["checkpoint"], map_location="cpu")
model.load_state_dict(state["model_state"])
model.eval()

# 4. Load audio
audio, sr = torchaudio.load("test_audio.wav")
if sr != 16000:
    audio = torchaudio.transforms.Resample(sr, 16000)(audio)
audio = audio.mean(dim=0)[:64600].unsqueeze(0)  # mono, pad/crop, add batch dim

# 5. Inference
with torch.no_grad():
    output = model(audio)
    score = output["scores"].item()

print(f"Score: {score:.4f}")
print(f"Prediction: {'bonafide' if score > 0 else 'spoof'}")
```

### Workflow 4: Compare models across datasets (Python)

```python
from deepfense.hub import list_models, download_model

# Find all WavLM + Nes2Net models trained on ASV19
models = list_models(pattern="ASV19_WavLM_Nes2Net")
print(models)
# ['ASV19_WavLM_Nes2Net_NoAug_Seed2',
#  'ASV19_WavLM_Nes2Net_NoAug_Seed42',
#  'ASV19_WavLM_Nes2Net_NoAug_Seed240']

# Download all three seeds for ensemble evaluation
for name in models:
    download_model(name, output_dir="./models")
```

---

## Dataset Format

The HuggingFace parquet files follow the same format DeepFense expects:

| Column | Type | Description |
|--------|------|-------------|
| `ID` | str | Unique sample identifier |
| `path` | str | Relative path to audio file |
| `label` | str | Label string (e.g. `"bonafide"`, `"spoof"`) |
| `dataset_name` | str | Dataset name |

**Important:** The `path` column contains *relative* paths. You need to set `root_dir` in your config to point to the directory where the actual audio files are stored.

---

## Model Naming Convention

```
{Dataset}_{Frontend}_{Backend}_{Augmentation}_{Seed}
```

| Part | Examples |
|------|----------|
| Dataset | `ASV19`, `CodecFake`, `FakeMusicCaps`, `HABLA` |
| Frontend | `WavLM`, `Wav2Vec2`, `EAT`, `mert` |
| Backend | `Nes2Net`, `AASIST`, `MLP`, `TCM` |
| Augmentation | `NoAug`, `RawBoost`, ... |
| Seed | `Seed2`, `Seed42`, `Seed240` |
