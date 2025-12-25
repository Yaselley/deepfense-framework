<div align="center">
  <img src="docs/logo/logo.png" alt="DeepFense Logo" width="400">
</div>

<div align="center">

# DeepFense Framework

**A Modular, Extensible Framework for Deepfake Audio Detection**

[![License](https://img.shields.io/badge/License-Apache%202.0-navy.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-navy.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-navy.svg)](https://pytorch.org/)

</div>

---

**DeepFense** is designed for researchers and developers to easily plug-and-play **Frontends** (Wav2Vec2, WavLM, HuBERT, EAT), **Backends** (AASIST, ECAPA-TDNN, RawNet2), and **Loss Functions** (OC-Softmax, AM-Softmax) to build state-of-the-art deepfake detection systems.

## ✨ Key Features

- 🔄 **Modular Architecture** — Swap frontends, backends, and losses with a single config change
- ⚙️ **Configuration-Driven** — All experiments defined in YAML
- 🎛️ **Advanced Augmentations** — RawBoost, RIR, Codec, Noise, and more
- 📊 **Built-in Metrics** — EER, minDCF, F1-score, Accuracy

---

## 🚀 Quick Start

### Installation

```bash
# From PyPI
pip install deepfense

# From source
git clone https://github.com/Yaselley/deepfense-framework
cd deepfense-framework
pip install -e .
```

### Train a Model

```bash
python train.py --config deepfense/config/train.yaml
```

### Test a Model

```bash
python test.py --config deepfense/config/train.yaml --checkpoint outputs/.../best_model.pth
```

---

## 📚 Documentation

| Guide | Description |
|-------|-------------|
| **[Installation](docs/01_installation.md)** | Full installation instructions |
| **[Quick Start](docs/02_quickstart.md)** | Train in 5 minutes |
| **[Full Tutorial](docs/03_full_tutorial.md)** | Complete config-driven training guide |
| **[Architecture](docs/04_architecture.md)** | How DeepFense works |
| **[Configuration](docs/05_configuration.md)** | All YAML parameters |
| **[Library Usage](docs/06_library_usage.md)** | Use DeepFense as a Python library |

### Component Reference

| Component | Examples |
|-----------|----------|
| [Frontends](docs/components/frontends.md) | Wav2Vec2, WavLM, HuBERT, MERT, EAT |
| [Backends](docs/components/backends.md) | AASIST, ECAPA-TDNN, RawNet2, MLP |
| [Losses](docs/components/losses.md) | OC-Softmax, AM-Softmax, A-Softmax, CrossEntropy |
| [Augmentations](docs/components/augmentations.md) | RawBoost, RIR, Noise, Codec, SpeedPerturb |

---

## 🏗️ Project Structure

```
deepfense-framework/
├── deepfense/
│   ├── config/          # YAML configurations
│   ├── data/            # Data handling & augmentations
│   ├── models/          # Frontends, backends, losses
│   ├── training/        # Training loop & evaluation
│   └── utils/           # Registry & helpers
├── docs/                # Documentation
├── train.py             # Training entry point
└── test.py              # Testing entry point
```

---

## 🔧 Extending DeepFense

Adding your own components is easy! DeepFense uses a registry pattern:

```python
# 1. Create and register your component
@register_backend("MyBackend")
class MyBackend(nn.Module):
    def __init__(self, config):
        ...

# 2. Use in YAML
backend:
  type: "MyBackend"
  args:
    param1: value1
```

See **[Extending DeepFense](docs/user_guide/extending.md)** for complete guides on adding:
- Custom Datasets
- Custom Frontends
- Custom Backends  
- Custom Loss Functions
- Custom Augmentations
- Custom Optimizers/Metrics

---

## 🎨 Contributing

We welcome contributions! See [Extending DeepFense](docs/user_guide/extending.md) for guidelines.

---

## 📄 License

Apache 2.0 — see [LICENSE](LICENSE) for details.

