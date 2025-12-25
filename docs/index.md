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

---

## 🧩 Component Reference

Detailed documentation for each module type:

| Component | Description |
|-----------|-------------|
| [Frontends](components/frontends.md) | Feature extractors: Wav2Vec2, WavLM, HuBERT, MERT, EAT |
| [Backends](components/backends.md) | Classifiers: AASIST, ECAPA-TDNN, RawNet2, MLP |
| [Losses](components/losses.md) | Loss functions: AM-Softmax, OC-Softmax, A-Softmax, CrossEntropy |
| [Augmentations](components/augmentations.md) | Data augmentation: RawBoost, RIR, Codec, Noise, etc. |
| [Optimizers & Schedulers](components/optimizers_schedulers.md) | Adam, SGD, CosineAnnealing, etc. |

---

## 🔧 User Guides

| Guide | Description |
|-------|-------------|
| [Training Workflow](user_guide/training.md) | Detailed training loop explanation |
| [Evaluation & Inference](user_guide/inference.md) | Testing and deployment |
| **[Extending DeepFense](user_guide/extending.md)** | **Add your own datasets, frontends, backends, losses, and more** |

---

## 🚀 Quick Links

- **[GitHub Repository](https://github.com/Yaselley/deepfense-framework)**
- **[Report Issues](https://github.com/Yaselley/deepfense-framework/issues)**

---

> **New to DeepFense?** Start with [Installation](01_installation.md) → [Quick Start](02_quickstart.md) → [Full Tutorial](03_full_tutorial.md)
