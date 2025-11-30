# DeepFense Project Structure

This document outlines the file structure of the DeepFense framework and the responsibility of each component.

```text
DeepFense/
├── README.md                       # Project overview, installation, and quick start guide
├── requirements.txt                # Python dependencies (torch, torchaudio, numpy, pandas, etc.)
├── setup.py                        # Package installation script
├── .gitignore                      # Git ignore rules (checkpoints, __pycache__, data files)
│
├── train.py                        # Main entry point for training
├── test.py                         # Main entry point for inference/testing
│
├── deepfense/                      # Main Package
│   ├── __init__.py
│   │
│   ├── config/                     # ⚙️ Configuration
│   │   ├── base.yaml               # Default base configuration
│   │   ├── train.yaml              # Main template for training experiments
│   │   └── examples/               # Specific experiment configurations
│   │       ├── augmentations_heavy.yaml
│   │       ├── augmentations_concat_all.yaml
│   │       ├── augmentations_no_pad.yaml
│   │       ├── augmentations_parallel.yaml
│   │       ├── augmentations_sequential.yaml
│   │       └── backend_mlp.yaml
│   │
│   ├── data/                       # 💾 Data Loading & Processing
│   │   ├── __init__.py
│   │   ├── base_dataset.py         # Abstract base class for datasets
│   │   ├── detection_dataset.py    # StandardDataset implementation (Parquet reading)
│   │   ├── data_utils.py           # Collate functions (padding, stacking) & DataLoader builders
│   │   └── transforms/             # Augmentations & Preprocessing
│   │       ├── __init__.py
│   │       ├── augmentations.py    # AugmentationPipeline, RIR, Noise, SpeedPerturb logic
│   │       ├── transforms.py       # Basic loading/cropping/padding
│   │       ├── utils.py            # Signal processing helpers (notch filter, alignment)
│   │       └── RawBoost/           # RawBoost algorithm implementation
│   │
│   ├── models/                     # 🧠 Model Architecture
│   │   ├── __init__.py
│   │   ├── base_model.py           # Base classes: BaseFrontend, BaseBackend, BaseLoss
│   │   ├── detector.py             # StandardDetector (The wrapper module connecting Front/Back/Loss)
│   │   │
│   │   ├── frontends/              # Feature Extractors (Waveform -> Features)
│   │   │   ├── __init__.py
│   │   │   ├── wav2vec2.py         # XLSR / Wav2Vec wrapper
│   │   │   ├── wavlm.py            # WavLM wrapper
│   │   │   └── hubert.py           # HuBERT wrapper
│   │   │
│   │   ├── backends/               # Classifiers/Embedders (Features -> Embeddings)
│   │   │   ├── __init__.py
│   │   │   ├── aasist.py           # AASIST graph network
│   │   │   ├── mlp.py              # Flexible MLP with pooling
│   │   │   ├── nes2net.py          # Res2Net-based backend
│   │   │   └── tcm.py              # Conformer-based backend
│   │   │
│   │   ├── losses/                 # Unified Loss Modules (Embedding -> Loss & Score)
│   │   │   ├── __init__.py
│   │   │   ├── cross_entropy.py    # Linear Projection + CE
│   │   │   ├── am_softmax.py       # Additive Margin Softmax
│   │   │   └── oc_softmax.py       # One-Class Softmax
│   │   │
│   │   └── modules/                # Reusable Neural Blocks
│   │       ├── pool.py             # Pooling factory (TAP, ASP, MHA)
│   │       └── conformer/          # Conformer blocks
│   │
│   ├── training/                   # 🏋️ Training Loop & Logic
│   │   ├── __init__.py
│   │   ├── base_trainer.py         # Abstract trainer (setup, logging, checkpointing)
│   │   ├── standard_trainer.py     # Main supervised training loop (handles 'concat' augs)
│   │   │
│   │   ├── evaluations/            # Metrics & Scoring
│   │   │   ├── evaluator.py        # Metric calculation orchestrator
│   │   │   ├── metrics.py          # EER, minDCF implementations
│   │   │   └── utils.py            # EER calculation helpers
│   │   │
│   │   ├── optimizers/             # Optimizer builders
│   │   │   └── utils.py
│   │   └── schedulers/             # Learning Rate Schedulers
│   │       └── utils.py
│   │
│   └── utils/                      # 🛠️ Utilities
│       ├── __init__.py
│       ├── registry.py             # The Core: dynamic class loading (@register_...)
│       ├── logger.py               # Logging setup
│       └── visualization.py        # (Optional) t-SNE plots, etc.
│
├── docs/                           # 📚 Documentation
│   ├── architecture.md
│   ├── components.md
│   ├── configuration.md
│   └── tutorials.md
│
└── outputs/                        # Experiment Artifacts (Ignored by git)
    └── Experiment_Name/
        ├── ckpts/                  # Saved models
        ├── logs/                   # Tensorboard/WandB logs
        └── results/                # Validation JSONs
```

### 🔑 Key Architectural Decisions

1.  **The Registry (`utils/registry.py`):**
    *   The backbone of the framework. It allows you to swap components in YAML (e.g., changing `backend: "AASIST"` to `backend: "MLP"`) without changing Python code.

2.  **Augmentation Pipeline (`data/transforms/augmentations.py`):**
    *   Now supports complex strategies: `Sequential` (Chain), `Parallel` (OneOf), and `Concat` (SpeechBrain-style branching) via a unified configuration.

3.  **Unified Loss Modules (`models/losses/`):**
    *   Classes like `AMSoftmax` contain *both* the trainable projection layer (fc weights) and the loss calculation logic. This simplifies the main model loop.

4.  **Modular Detector (`models/detector.py`):**
    *   The `StandardDetector` class acts as the glue. It instantiates the Frontend, Backend, and Loss based on the config, and manages the forward pass flow.

5.  **Standard Trainer (`training/standard_trainer.py`):**
    *   Contains the logic to handle special cases like the "Flattening" required for `concat` augmentations, decoupling the complex data shapes from the model architecture.

