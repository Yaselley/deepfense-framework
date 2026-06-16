# DeepFense Documentation

**DeepFense** is a modular, configuration-driven framework for deepfake audio detection. Mix and match frontends, backends, and loss functions via YAML -- no code changes needed.

!!! note "Partial deepfake / PartialSpoof"
    This version of the docs covers **clip-level** detection (one score per utterance). For per-frame labels, `TemporalDetector`, Range EER, and PartialSpoof metrics, use the **`deepfense-partial`** branch:

    ```bash
    git fetch origin
    git checkout deepfense-partial
    ```

    Documentation for that branch: [ReadTheDocs — deepfense-partial](https://deepfense.readthedocs.io/en/deepfense-partial/) (activate this version in ReadTheDocs if the link 404s).

---

## Getting Started

| Step | Guide | Description |
|------|-------|-------------|
| 1 | [Installation](01_installation.md) | Set up your environment |
| 2 | [Quick Start](02_quickstart.md) | Train your first model in 5 minutes |
| 3 | [Full Tutorial](03_full_tutorial.md) | Every config parameter explained |

---

## Reference

| Guide | Description |
|-------|-------------|
| [Architecture](04_architecture.md) | How DeepFense works internally |
| [Configuration](05_configuration.md) | All YAML parameters |
| [Library Usage](06_library_usage.md) | Use DeepFense as a Python library |
| [HuggingFace Hub](07_huggingface_hub.md) | Download datasets & pretrained models |
| [CLI Reference](cli_reference.md) | Command-line interface |
| [Pipeline Flow](pipeline_flow.md) | Complete data-to-evaluation pipeline |
| [Data Transforms](data_transforms.md) | All padding, cropping, and augmentation options |

---

## Component Reference

| Component | Description |
|-----------|-------------|
| [Frontends](components/frontends.md) | Wav2Vec2, WavLM, HuBERT, MERT, EAT |
| [Backends](components/backends.md) | AASIST, ECAPA-TDNN, Nes2Net, RawNet2, MLP, TCM |
| [Losses](components/losses.md) | CrossEntropy, OC-Softmax, AM-Softmax, A-Softmax |
| [Augmentations](components/augmentations.md) | RawBoost, RIR, Codec, Noise, SpeedPerturb, ... |
| [Optimizers & Schedulers](components/optimizers_schedulers.md) | Adam, SGD, CosineAnnealing, StepLR, ... |

---

## User Guides

| Guide | Description |
|-------|-------------|
| [Extending DeepFense](user_guide/extending.md) | Quick reference for all component types |
| [Adding Frontends](user_guide/adding_frontends.md) | Create custom feature extractors |
| [Adding Backends](user_guide/adding_backends.md) | Create custom classifiers |
| [Adding Losses](user_guide/adding_losses.md) | Create custom loss functions |
| [Adding Datasets](user_guide/adding_datasets.md) | Create custom datasets |
| [Adding Augmentations](user_guide/adding_augmentations.md) | Create custom augmentations |
| [Adding Optimizers](user_guide/adding_optimizers.md) | Add custom optimizers |
| [Adding Schedulers](user_guide/adding_schedulers.md) | Add custom schedulers |
| [Adding Metrics](user_guide/adding_metrics.md) | Add custom evaluation metrics |
| [Training Workflow](user_guide/training.md) | Detailed training loop explanation |
| [Training with CLI](user_guide/training_with_cli.md) | CLI-based training |
| [Inference](user_guide/inference.md) | Testing and deployment |

---

**New to DeepFense?** Start here: [Installation](01_installation.md) -> [Quick Start](02_quickstart.md) -> [Full Tutorial](03_full_tutorial.md)
