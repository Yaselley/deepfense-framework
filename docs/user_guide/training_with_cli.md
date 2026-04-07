# Training with the CLI

The `deepfense` CLI wraps the same logic as `train.py` and `test.py`. Use whichever you prefer.

## Train

```bash
deepfense train --config deepfense/config/train.yaml
```

Resume from a checkpoint:

```bash
deepfense train --config deepfense/config/train.yaml --resume outputs/exp/best_model.pth
```

| Option | Short | Required | Description |
|--------|-------|----------|-------------|
| `--config` | `-c` | Yes | Path to YAML config file |
| `--resume` | `-r` | No | Path to checkpoint to resume from |

## Test

```bash
deepfense test --config deepfense/config/train.yaml --checkpoint outputs/exp/best_model.pth
```

| Option | Short | Required | Description |
|--------|-------|----------|-------------|
| `--config` | `-c` | Yes | Path to YAML config file (must include `data.test` section) |
| `--checkpoint` | `-ckpt` | Yes | Path to `.pth` checkpoint |

## List Components

```bash
deepfense list                           # all components
deepfense list --component-type backends # only backends
```

## Equivalent Script Commands

| CLI | Script |
|-----|--------|
| `deepfense train -c config.yaml` | `python train.py --config config.yaml` |
| `deepfense test -c config.yaml -ckpt model.pth` | `python test.py --config config.yaml --checkpoint model.pth` |

## Typical Workflow

```bash
# 1. Generate test data
python tests/create_samples.py

# 2. See what components are available
deepfense list

# 3. Edit config (choose frontend, backend, loss)
#    deepfense/config/train.yaml

# 4. Train
deepfense train --config deepfense/config/train.yaml

# 5. Test
deepfense test \
    --config deepfense/config/train.yaml \
    --checkpoint outputs/*/best_model.pth
```

## Training Output

```
outputs/Wav2Vec2_Nes2Net_Example_20250401_120000/
├── config.yaml          # Saved config
├── train.log            # Full log
├── best_model.pth       # Best checkpoint
├── ckpts/               # All checkpoints
├── results/             # Per-epoch metrics
└── plots/               # Training curves
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `CUDA out of memory` | Reduce `batch_size`, set `freeze: True` on frontend |
| `Loss is NaN` | Reduce `lr` |
| `EER stuck at 50%` | Check data paths, try different `lr` |
| `deepfense: command not found` | Run `pip install -e .` from the repo root |

## Next Steps

- [Configuration Reference](../05_configuration.md) -- all YAML parameters
- [Extending DeepFense](extending.md) -- add custom components
- [Inference](inference.md) -- testing and evaluation details
