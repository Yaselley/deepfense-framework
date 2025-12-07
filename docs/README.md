# DeepFense Documentation

Welcome to the DeepFense framework documentation.

## Contents

### 1. Core Concepts
*   **[Architecture Overview](architecture.md)**
    *   Understand the ModularDetector, Registry, data flow diagrams, and project skeleton.

### 2. Tutorials
*   **[Getting Started & Configuration](tutorials/getting_started.md)**
    *   How to train your first model and a detailed explanation of all YAML parameters.
*   **[Extending DeepFense](tutorials/extending.md)**
    *   How to add new Frontends, Backends, Losses, and Datasets using the Registry system.

### 3. Component Reference
*   **[Frontends](components/frontends.md)**: Wav2Vec2, WavLM, HuBERT, MelSpectrogram.
*   **[Backends](components/backends.md)**: AASIST, MLP, Res2Net.
*   **[Loss Functions](components/losses.md)**: AM-Softmax, CrossEntropy, OC-Softmax, A-Softmax.
*   **[Data & Augmentations](components/data_pipeline.md)**: Parquet datasets, Augmentation pipelines (RawBoost, SpeedPerturb, etc.).
*   **[Optimizers & Schedulers](components/optimizers_schedulers.md)**: Adam, SGD, CosineAnnealing, StepLR.
