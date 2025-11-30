<div align="center">
  <img src="docs/logo/logo.png" alt="DeepFense Logo" width="400">
</div>

<div align="center">

# DeepFense Framework

**A Modular, Extensible Framework for Deepfake Audio Detection (ASV Spoofing)**

[![License](https://img.shields.io/badge/License-Apache%202.0-navy.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-navy.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-navy.svg)](https://pytorch.org/)

</div>

---

**DeepFense** is designed for researchers and developers to easily plug-and-play **Frontends** (Wav2Vec2, WavLM, EAT, MERT), **Backends** (AASIST, ECAPA-TDNN, RawNet2), and **Loss Functions** to build state-of-the-art deepfake detection systems.

## 📚 Documentation

The documentation is organized as follows:

### 🏠 Core
*   **[Architecture Overview](docs/architecture.md)**: **START HERE**. Explains the system skeleton, diagrams, and data flow.
*   **[Project Structure](docs/architecture.md#directory-structure-explained)**: Explanation of the file tree.

### 🎓 Tutorials
*   **[Getting Started & Training](docs/tutorials/getting_started.md)**: How to run training and a detailed guide to `config.yaml` parameters.
*   **[Extending DeepFense](docs/tutorials/extending.md)**: How to add new Losses, Frontends, Backends, and Datasets.

### 🧩 Components
Detailed reference for each module type:
*   **[Frontends](docs/components/frontends.md)**: Wav2Vec2, WavLM, HuBERT, MERT, EAT (Efficient Audio Transformer).
*   **[Backends](docs/components/backends.md)**: AASIST, ECAPA-TDNN, RawNet2, MLP, Nes2Net.
*   **[Loss Functions](docs/components/losses.md)**: AM-Softmax, A-Softmax, OC-Softmax, CrossEntropy.
*   **[Data & Augmentations](docs/components/data_pipeline.md)**: Parquet format, Augmentation pipelines (RawBoost, RIR, Codec, etc.).

## 🚀 Quick Start

1.  **Install Requirements**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Train a Model**
    ```bash
    python train.py --config deepfense/config/train.yaml
    ```

3.  **Evaluate/Test**
    ```bash
    python test.py --config deepfense/config/train.yaml --checkpoint outputs/.../best_model.pth
    ```

## 🎨 Contributing

We welcome contributions! Please see [Extending DeepFense](docs/tutorials/extending.md) for guidelines on how to add new components.

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
