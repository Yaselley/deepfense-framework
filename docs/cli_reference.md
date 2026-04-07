# CLI Reference

DeepFense provides a CLI for training, testing, and inspecting components.

## Setup

```bash
pip install -e .       # installs the `deepfense` command
deepfense --help       # verify it works
```

---

## Commands

### `deepfense train`

Train a model from a YAML config.

```bash
deepfense train --config deepfense/config/train.yaml
deepfense train -c deepfense/config/train.yaml --resume outputs/exp/best_model.pth
```

| Option | Short | Required | Description |
|--------|-------|----------|-------------|
| `--config` | `-c` | Yes | Path to YAML config file |
| `--resume` | `-r` | No | Resume from a checkpoint |

### `deepfense test`

Evaluate a trained model on the test set defined in the config.

```bash
deepfense test --config deepfense/config/train.yaml --checkpoint outputs/exp/best_model.pth
```

| Option | Short | Required | Description |
|--------|-------|----------|-------------|
| `--config` | `-c` | Yes | Path to YAML config file |
| `--checkpoint` | `-ckpt` | Yes | Path to `.pth` checkpoint |

### `deepfense list`

Show all registered components (frontends, backends, losses, etc.).

```bash
deepfense list                           # show everything
deepfense list --component-type backends # show only backends
```

| Option | Short | Default | Choices |
|--------|-------|---------|---------|
| `--component-type` | `-t` | `all` | `all`, `frontends`, `backends`, `losses`, `datasets`, `augmentations`, `optimizers`, `trainers` |

### `deepfense download`

Download datasets and pretrained models from [HuggingFace](https://huggingface.co/DeepFense).

```bash
# List available datasets
deepfense download list-datasets

# List available models (with optional filter)
deepfense download list-models
deepfense download list-models --filter WavLM --limit 50

# Download a dataset
deepfense download dataset CompSpoof
deepfense download dataset ASVSpoof19 --output-dir ./my_data

# Download a pretrained model
deepfense download model ASV19_WavLM_Nes2Net_NoAug_Seed42
deepfense download model ASV19_WavLM_Nes2Net_NoAug_Seed42 --output-dir ./my_models
```

See the [HuggingFace Hub Guide](07_huggingface_hub.md) for full usage details.

---

## Equivalent Python Scripts

The CLI wraps the same logic as the standalone scripts:

| CLI | Script |
|-----|--------|
| `deepfense train -c config.yaml` | `python train.py --config config.yaml` |
| `deepfense test -c config.yaml -ckpt model.pth` | `python test.py --config config.yaml --checkpoint model.pth` |

Both work identically. Use whichever you prefer.
