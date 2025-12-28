<div align="center">
  <img src="docs/logo/logo.png" alt="DeepFense Logo" width="400">
</div>

<div align="center">

# DeepFense Framework

**A Modular, Extensible Framework for Deepfake Audio Detection**

[![Website](https://img.shields.io/badge/🌐-Website-blue)](https://deepfense.github.io/)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow)](https://huggingface.co/DeepFense)
[![Documentation](https://img.shields.io/badge/📚-Documentation-green)](https://deepfense.readthedocs.io/)
[![Recipes](https://img.shields.io/badge/📖-Recipes-orange)](recipes/)

[![License](https://img.shields.io/badge/License-Apache%202.0-navy.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-navy.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-navy.svg)](https://pytorch.org/)

</div>

---

## What is DeepFense?

**DeepFense** is a modular framework for building and training deepfake audio detection models. It provides a plug-and-play architecture where you can easily combine different **Frontends** (feature extractors), **Backends** (classifiers), and **Loss Functions** to create state-of-the-art detection systems.

### Key Features

- 🔄 **Modular Architecture** — Swap components with a single config change
- ⚙️ **Configuration-Driven** — All experiments defined in YAML
- 🎛️ **Advanced Augmentations** — RawBoost, RIR, Codec, Noise, and more
- 📊 **Built-in Metrics** — EER, minDCF, F1-score, Accuracy
- 🚀 **Simple CLI** — Train and test models with command-line interface
- 📚 **Recipes** — Pre-configured training setups and example models ([see recipes](recipes/))

**New to DeepFense?** Check out our [recipes](recipes/) for pre-configured training setups and example models to get started quickly!

---

## Table of Contents

1. [Installation](#installation)
2. [Understanding DeepFense Architecture](#understanding-deepfense-architecture)
3. [Available Components](#available-components)
4. [Training Models](#training-models)
5. [Evaluating and Testing Models](#evaluating-and-testing-models)
6. [Data Preparation](#data-preparation)
7. [Extending DeepFense](#extending-deepfense)
8. [Using the CLI (Alternative)](#using-the-cli-alternative)
9. [Complete Pipeline Flow](#complete-pipeline-flow)
10. [Documentation](#documentation)

---

## Installation

```bash
# From source (recommended for development)
git clone https://github.com/Yaselley/deepfense-framework
cd deepfense-framework
pip install -e .

# Or From PyPI 
pip install deepfense
```

See [Installation Guide](docs/01_installation.md) for detailed instructions.

---

## Understanding DeepFense Architecture

DeepFense uses a **modular pipeline** architecture:

```
Raw Audio → Frontend → Features → Backend → Embeddings → Loss → Scores
```

**Key Components**:
- **Frontend**: Extracts features from audio (Wav2Vec2, WavLM, HuBERT, etc.)
- **Backend**: Processes features into embeddings (AASIST, ECAPA-TDNN, MLP, etc.)
- **Loss Function**: Computes loss and scores (CrossEntropy, OC-Softmax, etc.)

See [Architecture Overview](docs/04_architecture.md) for detailed architecture explanation, or [Pipeline Flow](docs/pipeline_flow.md) for complete pipeline walkthrough.

---

## Available Components

DeepFense provides a modular set of components that can be mixed and matched:

- **Frontends**: Wav2Vec2, WavLM, HuBERT, EAT, MERT - See [Frontends Documentation](docs/components/frontends.md)
- **Backends**: AASIST, ECAPA-TDNN, RawNet2, MLP, Nes2Net, TCM - See [Backends Documentation](docs/components/backends.md)
- **Losses**: CrossEntropy, OC-Softmax, AM-Softmax, A-Softmax - See [Losses Documentation](docs/components/losses.md)
- **Augmentations**: RawBoost, RIR, Codec, Noise, SpeedPerturb - See [Augmentations Documentation](docs/components/augmentations.md)

See [Component Reference](docs/index.md#-component-reference) for complete details.

**Looking for example configurations?** Check out our [recipes](recipes/) for pre-configured training setups and trained models.

---

## Training Models

Train models using Python scripts:

```bash
python train.py --config deepfense/config/train.yaml
```

Training creates an experiment directory with checkpoints, logs, and metrics.

**Alternative**: You can also use the CLI (see [Using the CLI](#using-the-cli) section below).

See [Quick Start Guide](docs/02_quickstart.md) for detailed instructions and [Configuration Reference](docs/05_configuration.md) for all YAML parameters.

---

## Evaluating and Testing Models

Test a trained model using Python scripts:

```bash
python test.py \
    --config deepfense/config/train.yaml \
    --checkpoint outputs/my_experiment/best_model.pth
```

DeepFense computes metrics automatically (EER, minDCF, F1, ACC) and saves results to the experiment directory.

**Alternative**: You can also use the CLI (see [Using the CLI](#using-the-cli) section below).

See [Evaluation & Inference Guide](docs/user_guide/inference.md) for details.

---

## Data Preparation

DeepFense uses **Parquet files** for dataset metadata. Each parquet file should contain:
- `ID`: Unique identifier
- `path`: Path to audio file
- `label`: Label string ("bonafide" or "spoof")
- `dataset_name`: (Optional) Dataset identifier

Example:
```python
import pandas as pd
data = pd.DataFrame({
    "ID": ["sample_001", "sample_002"],
    "path": ["/path/to/audio1.flac", "/path/to/audio2.flac"],
    "label": ["bonafide", "spoof"]
})
data.to_parquet("train.parquet")
```

### Data Transforms, Padding, and Cropping

DeepFense applies transforms in two stages:
1. **Base Transforms** (always): Padding, cropping, resampling
2. **Augmentations** (training only): RawBoost, RIR, Noise, etc.

**Critical**: All audio must be padded/cropped to the same length for batching. Configure this in your YAML:

```yaml
data:
  train:
    base_transform:
      - type: "pad"
        args:
          max_len: 160000       # 10 seconds @ 16kHz
          random_pad: True      # Random crop if longer
          pad_type: "repeat"    # Repeat if shorteror
```

**See [Data Transforms Guide](docs/data_transforms.md)** for complete transform parameters, padding/cropping options, augmentations, and how to check/modify configurations.

---

## Extending DeepFense

DeepFense makes it easy to add custom components using the registry pattern. Each component type has a detailed guide:

- [Adding Backends](docs/user_guide/adding_backends.md) | [Adding Frontends](docs/user_guide/adding_frontends.md) | [Adding Losses](docs/user_guide/adding_losses.md)
- [Adding Datasets](docs/user_guide/adding_datasets.md) | [Adding Augmentations](docs/user_guide/adding_augmentations.md)
- [Adding Optimizers](docs/user_guide/adding_optimizers.md) | [Adding Metrics](docs/user_guide/adding_metrics.md) | [Adding Schedulers](docs/user_guide/adding_schedulers.md)

See [Extending DeepFense (Quick Reference)](docs/user_guide/extending.md) for a quick overview of all component types.

---

## Using the CLI (Alternative)

DeepFense provides a CLI as an alternative to Python scripts. The CLI supports:

```bash
# Train a model (alternative to python train.py)
deepfense train --config config/train.yaml

# Test a model (alternative to python test.py)
deepfense test --config config/train.yaml --checkpoint outputs/exp/best_model.pth

# List available components
deepfense list
```

**Note**: The CLI currently supports training and testing existing models with different parameters. Future versions will support adding components via CLI.

See [CLI Reference](docs/cli_reference.md) for complete documentation.

---

## Complete Pipeline Flow

The DeepFense pipeline: **Data → Transforms → Frontend → Backend → Loss → Training → Evaluation**

See [Pipeline Flow Documentation](docs/pipeline_flow.md) for the complete detailed pipeline with all stages, data shapes, and configuration flow.

---

## Documentation

### Getting Started

| Guide | Description |
|-------|-------------|
| [Installation](docs/01_installation.md) | Full installation instructions |
| [Quick Start](docs/02_quickstart.md) | Train your first model in 5 minutes |
| [Full Tutorial](docs/03_full_tutorial.md) | Complete config-driven training guide |
| [Architecture](docs/04_architecture.md) | How DeepFense works internally |

### Component Reference

| Component | Documentation |
|-----------|--------------|
| [Frontends](docs/components/frontends.md) | Wav2Vec2, WavLM, HuBERT, MERT, EAT |
| [Backends](docs/components/backends.md) | AASIST, ECAPA_TDNN, RawNet2, MLP, Pool, Nes2Net, TCM |
| [Losses](docs/components/losses.md) | CrossEntropy, OC-Softmax, AM-Softmax, A-Softmax |
| [Augmentations](docs/components/augmentations.md) | RawBoost, RIR, Codec, Noise, SpeedPerturb |
| [Optimizers & Schedulers](docs/components/optimizers_schedulers.md) | Adam, SGD, CosineAnnealing, StepLR |

### User Guides

| Guide | Description |
|-------|-------------|
| [Training with CLI](docs/user_guide/training_with_cli.md) | How to train models using the CLI |
| [Training Workflow](docs/user_guide/training.md) | Detailed training loop explanation |
| [Evaluation & Inference](docs/user_guide/inference.md) | Testing and deployment |
| [Configuration Reference](docs/05_configuration.md) | All YAML parameters explained |
| [Library Usage](docs/06_library_usage.md) | Use DeepFense programmatically in Python |

### Extending DeepFense

| Guide | Description |
|-------|-------------|
| [Adding a New Backend](docs/user_guide/adding_backends.md) | Step-by-step guide to create custom backends |
| [Adding a New Frontend](docs/user_guide/adding_frontends.md) | Step-by-step guide to create custom frontends |
| [Adding a New Loss](docs/user_guide/adding_losses.md) | Step-by-step guide to create custom loss functions |
| [Adding a New Dataset](docs/user_guide/adding_datasets.md) | Step-by-step guide to create custom datasets |
| [Adding Augmentations](docs/user_guide/adding_augmentations.md) | Step-by-step guide to create custom data augmentations |
| [Adding Optimizers](docs/user_guide/adding_optimizers.md) | Step-by-step guide to add custom optimizers |
| [Adding Metrics](docs/user_guide/adding_metrics.md) | Step-by-step guide to add custom evaluation metrics |
| [Adding Schedulers](docs/user_guide/adding_schedulers.md) | Step-by-step guide to add custom learning rate schedulers |
| [Extending DeepFense (Quick Reference)](docs/user_guide/extending.md) | Quick reference for all component types |

### CLI Reference

| Guide | Description |
|-------|-------------|
| [CLI Reference](docs/cli_reference.md) | Complete CLI documentation |

### Recipes

| Resource | Description |
|----------|-------------|
| [Recipes](recipes/) | Pre-configured training setups and example models |

---

## Project Structure

```
deepfense/
├── config/          # YAML configurations
├── data/            # Data handling & transforms
├── models/          # Frontends, backends, losses
├── training/        # Training loop & evaluation
├── utils/           # Registry & helpers
└── cli/             # Command-line interface
```

See [Architecture Overview](docs/04_architecture.md) for detailed structure and component organization.

---

## Recipes

DeepFense provides example recipes (pre-configured training setups) to help you get started quickly. Each recipe includes:
- Complete configuration files
- Pre-trained model checkpoints (where available)
- Training scripts and evaluation results
- Documentation on architecture choices and hyperparameters

See the [recipes](recipes/) folder for available recipes. Each recipe includes detailed README files explaining the configuration and how to reproduce the results.

---

## Contributing

We welcome contributions! See [Extending DeepFense](docs/user_guide/extending.md) for guidelines on adding new components.

---

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.

---

## Citation

If you use DeepFense in your research, please cite:

```bibtex
@software{deepfense2024,
  title={DeepFense: A Modular Framework for Deepfake Audio Detection},
  author={DeepFense Team},
  year={2024},
  url={https://github.com/Yaselley/deepfense-framework}
}
```
