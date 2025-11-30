# DeepFense Documentation

Welcome to the DeepFense framework documentation.

## Contents

1.  **[Architecture Overview](architecture.md)**
    *   Understand the ModularDetector, Registry, and Unified Loss design.
2.  **[Component Reference](components.md)**
    *   Detailed list of available Frontends, Backends, Losses, and Metrics.
3.  **[Configuration Guide](configuration.md)**
    *   How to structure your YAML experiment files.
4.  **[Extending DeepFense](extending.md)**
    *   **Base Classes** and guidelines for creating new Frontends, Backends, and Losses.
5.  **[Tutorials](tutorials.md)**
    *   Step-by-step guides for common tasks.

## Quick Start

**Train a model:**
```bash
python train.py --config deepfense/config/train.yaml
```

**Project Structure:**
*   `deepfense/models/`: Frontends, Backends, Detectors, Losses.
*   `deepfense/data/`: Datasets, Transforms.
*   `deepfense/training/`: Trainer loops, Evaluators, Metrics.
*   `deepfense/utils/`: Registry, logging, etc.
