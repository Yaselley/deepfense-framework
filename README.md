# DeepFense Framework

**DeepFense** is a modular, extensible, and research-friendly framework for Deepfake Audio Detection (ASV Spoofing). It allows you to easily mix and match Frontends, Backends, and Loss functions using a simple configuration system.

## 🚀 Features

*   **Modular Architecture**: Decouples **Frontends** (Wav2Vec2, WavLM, HuBERT), **Backends** (AASIST, Nes2Net, TCM, MLP), and **Losses**.
*   **Unified Registry System**: Easily register and call components via YAML strings (e.g., `type: "wavlm"`).
*   **Unified Loss Modules**: "Mappers" (projection layers) and Loss functions are combined into single modules for cleaner code.
*   **Base Classes**: Standardized `BaseFrontend`, `BaseBackend`, and `BaseLoss` classes guide extension and ensure consistency.
*   **Loss-Dependent Scoring**: Automatically handles scoring logic (Logits vs. Cosine Similarity vs. LLR) for correct EER/minDCF calculation.
*   **Configurable**: Powered by `OmegaConf` and YAML for hierarchical configuration.

## 📚 Documentation

Detailed documentation is available in the `docs/` folder:

1.  **[Architecture Overview](docs/architecture.md)**: High-level design and data flow.
2.  **[Component Reference](docs/components.md)**: List of available models and losses.
3.  **[Configuration Guide](docs/configuration.md)**: How to write your experiment YAMLs.
4.  **[Tutorials](docs/tutorials.md)**: Step-by-step guides to adding new components.
5.  **[Extending DeepFense](docs/extending.md)**: Deep dive into Base Classes and API.

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-repo/DeepFense.git
    cd DeepFense
    ```

2.  **Install dependencies**:
    It is recommended to use a virtual environment (Conda or venv).
    ```bash
    pip install -r requirements.txt
    ```

## 🚦 Quick Start

### 1. Training

To start a training experiment, use the `train.py` script with a configuration file.

```bash
python train.py --config deepfense/config/train.yaml
```

**Key Config Sections:**
*   **`model`**: Defines the architecture (Frontend + Backend + Loss).
*   **`data`**: Defines training/validation datasets and augmentations.
*   **`training`**: Defines optimizer, scheduler, and evaluation metrics.

### 2. Testing / Inference

To evaluate a trained checkpoint:

```bash
python test.py --config deepfense/config/train.yaml --checkpoint outputs/ExperimentName/checkpoints/best_model.pt
```

This will generate a `results.json` and prediction files in the output directory.

## 📂 Project Structure

```text
DeepFense/
├── deepfense/
│   ├── config/           # YAML Configuration files
│   ├── data/             # Datasets and Transforms
│   ├── models/           # Core Model Components
│   │   ├── backends/     # (aasist.py, mlp.py, etc.)
│   │   ├── frontends/    # (wavlm.py, hubert.py, etc.)
│   │   ├── losses/       # (am_softmax.py, cross_entropy.py, etc.)
│   │   ├── detector.py   # Main ModularDetector class
│   │   └── base_model.py # Abstract Base Classes
│   ├── training/         # Trainer, Evaluator, Metrics
│   └── utils/            # Registry, Logging, Helper functions
├── docs/                 # Documentation
├── outputs/              # Experiment logs and checkpoints
├── train.py              # Training entry point
├── test.py               # Testing entry point
└── requirements.txt      # Dependencies
```

## 🧩 Example Configuration

Here is a minimal example of a model definition in `train.yaml`:

```yaml
model:
  type: "StandardDetector"
  
  # 1. Frontend (Audio -> Features)
  frontend:
    type: "wavlm"
    args:
      ckpt_path: "/path/to/wavlm.pt"

  # 2. Backend (Features -> Embedding)
  backend:
    type: "AASIST"
    args:
      filts: [[1, 32], [32, 32]]
      gat_dims: [64, 32]

  # 3. Loss (Embedding -> Loss & Score)
  loss:
    - type: "AMSoftmax"
      weight: 1.0
      embedding_dim: 128
      n_classes: 2
      m: 0.3
      s: 30
```

## 🤝 Contributing

We welcome contributions! Please refer to the **[Tutorials](docs/tutorials.md)** to see how to easily add new Frontends, Backends, or Metrics using our standardized Base Classes.

## 📄 License

[License Information Here]
