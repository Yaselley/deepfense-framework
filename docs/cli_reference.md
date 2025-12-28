# DeepFense CLI Reference

The DeepFense CLI provides commands to train models, test models, add components, generate data, and more.

## Installation

After installing DeepFense, the CLI is available as the `deepfense` command:

```bash
pip install -e .  # For development
# or
pip install deepfense  # From PyPI (when published)
```

## Commands Overview

```bash
deepfense --help
```

### Training

Train a model using a configuration file:

```bash
deepfense train --config deepfense/config/train.yaml
```

Resume training from a checkpoint:

```bash
deepfense train --config deepfense/config/train.yaml --resume outputs/exp/best_model.pth
```

### Testing

Test a trained model:

```bash
deepfense test --config deepfense/config/train.yaml --checkpoint outputs/exp/best_model.pth
```

### Adding Components

To add new components, see the detailed tutorials:
- [Adding a New Backend](user_guide/adding_backends.md)
- [Adding a New Frontend](user_guide/adding_frontends.md)
- [Extending DeepFense](user_guide/extending.md) - for datasets, losses, augmentations, optimizers

### Generate Data

Generate parquet files from protocol files:

```bash
deepfense generate-data deepfense/config/parquets/generate_asv19.py
```

This runs the specified data generation script.

### List Components

List all available components:

```bash
deepfense list
```

List components of a specific type:

```bash
deepfense list --component-type backends
deepfense list --component-type frontends
deepfense list --component-type losses
deepfense list --component-type datasets
deepfense list --component-type augmentations
deepfense list --component-type optimizers
deepfense list --component-type trainers
```

## Examples

### Complete Workflow

1. **List available components:**
   ```bash
   deepfense list
   ```

2. **Generate training data:**
   ```bash
   deepfense generate-data deepfense/config/parquets/generate_asv19.py
   ```

3. **Add a custom backend** (see [Adding a New Backend](user_guide/adding_backends.md)):
   - Create the backend file following the tutorial
   - Import it in `__init__.py`
   - Register it with `@register_backend`

4. **Train a model:**
   ```bash
   deepfense train --config deepfense/config/train.yaml
   ```

5. **Test the model:**
   ```bash
   deepfense test --config deepfense/config/train.yaml --checkpoint outputs/exp/best_model.pth
   ```

## Command Reference

### `deepfense train`

Train a DeepFense model.

**Options:**
- `--config, -c`: Path to YAML config file (required)
- `--resume, -r`: Resume from checkpoint (optional)

### `deepfense test`

Test a trained DeepFense model.

**Options:**
- `--config, -c`: Path to YAML config file (required)
- `--checkpoint, -ckpt`: Path to model checkpoint file (required)


### `deepfense generate-data`

Generate parquet data files from protocol files.

**Arguments:**
- `script_path`: Path to the data generation Python script

### `deepfense list`

List all available components in DeepFense.

**Options:**
- `--component-type, -t`: Type of component to list (default: all)
  - Options: `all`, `frontends`, `backends`, `losses`, `datasets`, `augmentations`, `optimizers`, `trainers`

## Tips

1. **Use `deepfense list`** to see all available components
2. **Follow the tutorials** for adding new components:
   - [Adding a New Backend](user_guide/adding_backends.md)
   - [Adding a New Frontend](user_guide/adding_frontends.md)
   - [Extending DeepFense](user_guide/extending.md) for other components
3. **Test your components** with dummy data before using in training
4. **Import new components** in their respective `__init__.py` files after creation

