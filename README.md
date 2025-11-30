# DeepFense Framework

**DeepFense** is a modular, extensible framework for Deepfake Audio Detection (ASV Spoofing). It allows researchers to easily plug-and-play Frontends, Backends, and Loss functions.

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
*   **[Frontends](docs/components/frontends.md)** (Wav2Vec2, WavLM, HuBERT, etc.)
*   **[Backends](docs/components/backends.md)** (AASIST, MLP, etc.)
*   **[Loss Functions](docs/components/losses.md)** (AM-Softmax, CrossEntropy, etc.)
*   **[Data & Augmentations](docs/components/data_pipeline.md)** (Parquet format, Augmentation pipelines).

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
