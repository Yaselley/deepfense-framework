# Welcome to DeepFense

**DeepFense** is a modular, configuration-driven framework designed for **Deepfake Audio Detection**. It decouples the **Frontend** (feature extraction), **Backend** (classification), and **Loss** functions, allowing researchers to mix and match components easily via YAML configuration files.

## Key Features

*   **Modular Design**: Swap Wav2Vec2 for WavLM or AASIST for ResNet with a single config change.
*   **Configuration-Driven**: All hyperparameters (learning rates, augmentations, model architectures) are defined in `config/train.yaml`.
*   **Advanced Augmentations**: Includes a powerful pipeline for RawBoost, RIR (Reverb), Codec simulation, and more.
*   **Standardized Training**: Built-in support for EER (Equal Error Rate) and minDCF metrics, logging to WandB, and checkpoint management.

## Quick Links

*   [Getting Started](tutorials/getting_started.md): Install and run your first experiment.
*   [Architecture](architecture.md): Understand how data flows through the system.
*   [Configuration](user_guide/configuration.md): Learn how to customize experiments.
*   [API Reference](api/models/detector.md): Deep dive into the code.

