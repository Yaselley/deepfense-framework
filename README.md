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
*   **[Data & Augmentations](docs/components/data_pipeline.md)**: Parquet format, Augmentation pipelines (RawBoost, RIR, Codec, etc.).
*   **[Frontends](docs/components/frontends.md)**: Wav2Vec2, WavLM, HuBERT, MERT, EAT (Efficient Audio Transformer).
*   **[Backends](docs/components/backends.md)**: AASIST, ECAPA-TDNN, RawNet2, MLP, Nes2Net.
*   **[Loss Functions](docs/components/losses.md)**: AM-Softmax, A-Softmax, OC-Softmax, CrossEntropy.

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Yaselley/deepfense-framework
    cd DeepFense
    ```

2.  **Install dependencies**:
    It is recommended to use a virtual environment (Conda or venv).
    
    **Step A: Install core requirements**
    ```bash
    pip install -r requirements.txt
    ```

    **Step B: Downgrade pip (Required for Fairseq)**
    Some dependencies require an older pip version to build correctly.
    ```bash
    pip install "pip<=24.0"
    ```

    **Step C: Build Fairseq**
    DeepFense relies on Fairseq for SSL model integration.
    ```bash
    mkdir -p deepfense/models/modules
    cd deepfense/models/modules
    git clone https://github.com/facebookresearch/fairseq
    cd fairseq
    git checkout 3d262bb
    pip install --editable ./
    
    # Optional: Upgrade pip back if needed
    # pip install --upgrade pip
    cd ../../../..
    ```

## 🚀 Quick Start

1.  **Train a Model**
    ```bash
    python train.py --config deepfense/config/train.yaml
    ```

2.  **Evaluate/Test**
    ```bash
    python test.py --config deepfense/config/train.yaml --checkpoint outputs/.../best_model.pth
    ```

## 🎨 Contributing

We welcome contributions! Please see [Extending DeepFense](docs/tutorials/extending.md) for guidelines on how to add new components.

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
