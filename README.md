DeepFense/
├── README.md
├── LICENSE
├── setup.py                     # or pyproject.toml if using Poetry
├── requirements.txt             # minimal dependencies
├── .gitignore
├── .github/
│   ├── workflows/
│   │   └── ci.yml              # continuous integration tests (lint, format, unit tests)
│   └── ISSUE_TEMPLATE/
│       ├── bug_report.md
│       └── feature_request.md
│
├── deepfense/                   # main Python package
│   ├── __init__.py
│   │
│   ├── config/                  # experiment and system configs
│   │   ├── base.yaml
│   │   ├── model/
│   │   │   ├── wavlm.yaml
│   │   │   ├── wav2vec.yaml
│   │   │   └── ...
│   │   ├── dataset/
│   │   │   ├── asvspoof.yaml
│   │   │   ├── add23.yaml
│   │   │   └── ...
│   │   └── training/
│   │       ├── default.yaml
│   │       ├── lr_scheduler.yaml
│   │       └── augmentation.yaml
│   │
│   ├── data/                    # dataset loaders & preprocessing
│   │   ├── __init__.py
│   │   ├── base_dataset.py
│   │   ├── asvspoof_loader.py
│   │   ├── add23_loader.py
│   │   ├── fakeavceleb_loader.py
│   │   └── transforms/          # online & offline augmentations
│   │       ├── __init__.py
│   │       ├── noise.py
│   │       ├── speed.py
│   │       └── pitch.py
│   │
│   ├── models/                  # model architectures
│   │   ├── __init__.py
│   │   ├── backbones/
│   │   │   ├── wavlm.py
│   │   │   ├── wav2vec2.py
│   │   │   ├── ecapa.py
│   │   │   └── cnn_baseline.py
│   │   ├── heads/
│   │   │   ├── classifier_head.py
│   │   │   ├── source_head.py
│   │   │   └── multi_task_head.py
│   │   └── utils.py
│   │
│   ├── training/                # training and evaluation logic
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── evaluator.py
│   │   ├── losses.py
│   │   ├── optimizers.py
│   │   ├── schedulers.py
│   │   └── callbacks.py
│   │
│   ├── eval/                    # evaluation & metrics
│   │   ├── __init__.py
│   │   ├── metrics.py           # EER, AUC, DET, attribution acc, etc.
│   │   ├── plot_curves.py       # ROC/DET plotting
│   │   └── benchmark.py         # full evaluation suite
│   │
│   ├── utils/                   # logging, seeding, configs, misc
│   │   ├── __init__.py
│   │   ├── logger.py
│   │   ├── config_utils.py
│   │   ├── seed.py
│   │   ├── paths.py
│   │   └── visualization.py
│   │
│   └── cli/                     # command-line interfaces
│       ├── __init__.py
│       ├── train.py             # python -m deepfense.cli.train
│       ├── evaluate.py
│       ├── benchmark.py
│       └── trace.py             # source attribution pipeline
│
├── experiments/                 # ready-to-run experiment configs
│   ├── baseline/
│   │   ├── wavlm_small.yaml
│   │   ├── wav2vec_small.yaml
│   │   └── readme.md
│   ├── ablation/
│   │   ├── layerwise.yaml
│   │   └── augmentation.yaml
│   └── ...
│
├── scripts/                     # helper scripts
│   ├── run_train.sh
│   ├── run_eval.sh
│   ├── run_benchmark.sh
│   ├── prepare_dataset.sh
│   └── download_models.py
│
├── notebooks/                   # Jupyter demos / analysis
│   ├── demo_detection.ipynb
│   ├── feature_visualization.ipynb
│   └── attribution_examples.ipynb
│
├── tests/                       # unit & integration tests
│   ├── __init__.py
│   ├── test_datasets.py
│   ├── test_models.py
│   ├── test_trainer.py
│   ├── test_metrics.py
│   └── test_cli.py
│
└── docs/                        # documentation
    ├── index.md
    ├── quickstart.md
    ├── architecture.md
    ├── datasets.md
    ├── configs.md
    └── contributing.md


⚙️ Structure Overview
| Folder                | Purpose                                                     |
| :-------------------- | :---------------------------------------------------------- |
| `deepfense/`          | Core framework package (importable as `deepfense`)          |
| `deepfense/data/`     | Dataset loaders and augmentations                           |
| `deepfense/models/`   | Model backbones (SSL or CNN) and classifier heads           |
| `deepfense/training/` | Training and evaluation loops, losses, optimizers           |
| `deepfense/eval/`     | Metric computation and benchmarking utilities               |
| `deepfense/utils/`    | Helper utilities: logging, seeding, config, visualization   |
| `deepfense/cli/`      | Command-line entry points for training, evaluation, tracing |
| `experiments/`        | Organized config files for reproducible experiments         |
| `scripts/`            | Shell scripts for running experiments or preparing data     |
| `notebooks/`          | Demonstrations, visualizations, exploratory analysis        |
| `tests/`              | Unit and integration tests for CI                           |
| `docs/`               | Project documentation (Markdown or Sphinx)                  |


🧪 Example CLI Usage
After installing the package locally:
python -m deepfense.cli.train --config experiments/baseline/wavlm_small.yaml
python -m deepfense.cli.evaluate --checkpoint outputs/wavlm_small.ckpt
python -m deepfense.cli.trace --audio_path samples/fake.wav
