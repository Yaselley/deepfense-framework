# DeepFense Documentation

Welcome to the **DeepFense Framework** documentation — your comprehensive guide to building state-of-the-art deepfake audio detection systems.

---

## 📖 How to Read This Documentation

This documentation is organized as a **progressive learning path**. We recommend reading in order:

| Chapter | Title | Description |
|---------|-------|-------------|
| **01** | [Installation](01_installation.md) | Set up your environment |
| **02** | [Quick Start](02_quickstart.md) | Train your first model in 5 minutes |
| **03** | [Complete Training Tutorial](03_full_tutorial.md) | Config-driven training guide |
| **04** | [Architecture Overview](04_architecture.md) | Understand how DeepFense works |
| **05** | [Configuration Reference](05_configuration.md) | All YAML parameters explained |
| **06** | [Library Usage](06_library_usage.md) | Use DeepFense programmatically in Python |
| **07** | [CLI Reference](cli_reference.md) | Command-line interface documentation |
| **08** | [Pipeline Flow](pipeline_flow.md) | Complete pipeline from data to deployment |
| **[Data Transforms](data_transforms.md)** | **All transform options, padding, cropping, and augmentation parameters** |
| **[Recipes](../recipes/)** | **Pre-configured training setups and example models** |

---

## 🧩 Component Reference

Detailed documentation for each module type:

| Component | Description |
|-----------|-------------|
| [Frontends](components/frontends.md) | Feature extractors: Wav2Vec2, WavLM, HuBERT, MERT, EAT |
| [Backends](components/backends.md) | Classifiers: AASIST, ECAPA_TDNN, RawNet2, MLP, Pool, Nes2Net, TCM |
| [Losses](components/losses.md) | Loss functions: AM-Softmax, OC-Softmax, A-Softmax, CrossEntropy |
| [Augmentations](components/augmentations.md) | Data augmentation: RawBoost, RIR, Codec, Morph, AdditiveNoise, SpeedPerturb, AddBabble, DropFreq, DropChunk, DoClip |
| [Optimizers & Schedulers](components/optimizers_schedulers.md) | Adam, SGD, CosineAnnealing, etc. |

---

## 🔧 User Guides

| Guide | Description |
|-------|-------------|
| [Training Workflow](user_guide/training.md) | Detailed training loop explanation |
| [Training with CLI](user_guide/training_with_cli.md) | How to train models using the Library CLI |
| [Evaluation & Inference](user_guide/inference.md) | Testing and deployment |
| [Adding a New Backend](user_guide/adding_backends.md) | Step-by-step guide to create custom backends |
| [Adding a New Frontend](user_guide/adding_frontends.md) | Step-by-step guide to create custom frontends |
| [Adding a New Loss](user_guide/adding_losses.md) | Step-by-step guide to create custom loss functions |
| [Adding a New Dataset](user_guide/adding_datasets.md) | Step-by-step guide to create custom datasets |
| [Adding Augmentations](user_guide/adding_augmentations.md) | Step-by-step guide to create custom data augmentations |
| [Adding Optimizers](user_guide/adding_optimizers.md) | Step-by-step guide to add custom optimizers |
| [Adding Metrics](user_guide/adding_metrics.md) | Step-by-step guide to add custom evaluation metrics |
| [Adding Schedulers](user_guide/adding_schedulers.md) | Step-by-step guide to add custom learning rate schedulers |
| **[Extending DeepFense](user_guide/extending.md)** | **Quick reference for all component types** |

---

## 🚀 Quick Links

- **[GitHub Repository](https://github.com/Yaselley/deepfense-framework)**
- **[Report Issues](https://github.com/Yaselley/deepfense-framework/issues)**

---

> **New to DeepFense?** Start with [Installation](01_installation.md) → [Quick Start](02_quickstart.md) → [Full Tutorial](03_full_tutorial.md)
