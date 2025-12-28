# Training with DeepFense CLI

This guide shows you how to train models using the DeepFense command-line interface.

## Quick Start

The simplest way to train a model:

```bash
deepfense train --config deepfense/config/train.yaml
```

## Command Options

```bash
deepfense train --help
```

### Required Arguments

- `--config, -c`: Path to YAML configuration file

### Optional Arguments

- `--resume, -r`: Path to checkpoint file to resume training from

## Configuration File Structure

Your configuration file should follow this structure:

```yaml
seed: 42

data:
  sampling_rate: 16000
  label_map:
    bonafide: 1
    spoof: 0

  train:
    dataset_type: "DetectionDataset"
    parquet_files: ["/path/to/train.parquet"]
    label_map: ${data.label_map}
    target_sr: ${data.sampling_rate}
    mono: True
    batch_size: 8
    shuffle: True
    base_transform: []
    augment_transform:
      - type: "RandomCrop"
        args: {output_size: 160000}
      - type: "AdditiveNoise"
        args: {snr_range: [5, 15]}

  val:
    dataset_type: "DetectionDataset"
    parquet_files: ["/path/to/val.parquet"]
    label_map: ${data.label_map}
    target_sr: ${data.sampling_rate}
    mono: True
    batch_size: 8
    shuffle: False
    base_transform: []
    augment_transform: []

model:
  type: "Detector"
  frontend:
    type: "wav2vec2"
    args:
      ckpt_path: "/path/to/wav2vec2.pt"
      freeze: True
      output_dim: 768
  backend:
    type: "AASIST"
    args:
      input_dim: 768
      output_dim: 128
  loss:
    - type: "CrossEntropyLoss"
      args: {}

training:
  type: "StandardTrainer"
  exp_name: "my_experiment"
  output_dir: "outputs"
  optimizer:
    type: "Adam"
    args:
      lr: 0.0001
  scheduler:
    type: "CosineAnnealingLR"
    args:
      T_max: 100
  epochs: 100
  eval_every_epochs: 1
  monitor_metric: "eer"
  monitor_mode: "min"
  wandb: True
  wandb_project: "DeepFense"
  metrics:
    ACC: {}
    F1_SCORE: {}
    EER: {}
    minDCF:
      Pspoof: 0.05
      Cmiss: 1
      Cfa: 1
```

## Training Examples

### Basic Training

```bash
deepfense train --config configs/my_config.yaml
```

### Resume Training

```bash
deepfense train \
    --config configs/my_config.yaml \
    --resume outputs/my_experiment_20241228_120000/best_model.pth
```

### Training with Custom Config

```bash
deepfense train --config configs/custom_experiment.yaml
```

## Training Output

Training creates an experiment directory in your `output_dir`:

```
outputs/
└── my_experiment_20241228_120000/
    ├── config.yaml          # Saved configuration
    ├── train.log            # Training logs
    ├── best_model.pth       # Best model checkpoint
    ├── latest_model.pth     # Latest checkpoint
    ├── metrics.json         # Training metrics
    └── wandb/               # WandB logs (if enabled)
```

## Monitoring Training

### Logs

Training logs are written to `output_dir/exp_name_timestamp/train.log`:

```bash
tail -f outputs/my_experiment_20241228_120000/train.log
```

### WandB (if enabled)

If `wandb: True` in your config, training metrics are logged to WandB:

```python
# View at: https://wandb.ai/your-project/DeepFense
```

### Metrics

Key metrics are logged at each validation step:
- Loss
- Accuracy (ACC)
- F1 Score
- EER (Equal Error Rate)
- minDCF (minimum Detection Cost Function)

## Common Training Workflows

### Workflow 1: Train from Scratch

1. **Prepare data**: Create parquet files for train and validation sets
2. **Create config**: Write YAML configuration file
3. **Train**: Run `deepfense train --config config.yaml`
4. **Monitor**: Watch logs or WandB dashboard
5. **Test**: Use trained model with `deepfense test`

### Workflow 2: Resume Interrupted Training

1. **Identify checkpoint**: Find the latest checkpoint in outputs directory
2. **Resume**: Run `deepfense train --config config.yaml --resume checkpoint.pth`
3. **Continue**: Training resumes from the checkpoint epoch

### Workflow 3: Fine-tune Pretrained Model

1. **Load pretrained**: Set `ckpt_path` in frontend config
2. **Freeze/unfreeze**: Control which layers to train
3. **Lower learning rate**: Use smaller LR for fine-tuning
4. **Train**: Run training command

## Troubleshooting

### Out of Memory

- Reduce `batch_size` in data config
- Use gradient accumulation (if supported)
- Reduce model size

### Training Too Slow

- Increase `batch_size` if memory allows
- Use fewer augmentations
- Reduce validation frequency (`eval_every_epochs`)

### Poor Performance

- Check data quality and labels
- Try different augmentations
- Adjust learning rate
- Try different backends/frontends
- Increase training epochs

### Checkpoint Issues

- Ensure checkpoint path is correct
- Verify checkpoint matches model architecture
- Check that config matches original training config

## Advanced Usage

### Multi-GPU Training

DeepFense supports multi-GPU training through PyTorch's DataParallel or DistributedDataParallel. Set `device: "cuda"` in training config and use:

```bash
CUDA_VISIBLE_DEVICES=0,1 deepfense train --config config.yaml
```

### Custom Output Directory

Set in config:
```yaml
training:
  output_dir: "/path/to/outputs"
```

Or override in command (if supported):
```bash
deepfense train --config config.yaml --output-dir /custom/path
```

## Next Steps

- See [Testing Guide](inference.md) for model evaluation
- See [Configuration Reference](../05_configuration.md) for all config options
- See [Adding Components](extending.md) for custom backends/frontends

