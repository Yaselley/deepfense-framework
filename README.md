# DeepFense Framework

DeepFense is a modular and extensible framework for training and evaluating deepfake detection models. It supports various frontends (SSL models like Wav2Vec2, HuBERT, WavLM), backends (AASIST, Nes2Net, TCM, MLP), and loss functions (CrossEntropy, A-Softmax, AM-Softmax, OC-Softmax).

## Features

*   **Modular Design**: Easily swap Frontends, Backends, and Losses via a config file.
*   **Unified Registry**: Simple registration mechanism for new components.
*   **Unified Loss/Mapper**: Loss functions handle their own projection layers (mappers), simplifying usage.
*   **Structured Configuration**: Clean YAML-based configuration.
*   **Extensible**: Easy to add new datasets, transforms, and metrics.

## Installation

```bash
pip install -r requirements.txt
```

## Directory Structure

*   `deepfense/models`: Contains Frontends, Backends, and Losses.
    *   `frontends/`: Wav2Vec2, WavLM, HuBERT, etc.
    *   `backends/`: AASIST, Nes2Net, TCM, etc.
    *   `losses/`: CrossEntropy, AMSoftmax, OCSoftmax, etc.
*   `deepfense/data`: Dataset and Transform logic.
*   `deepfense/training`: Trainer, Optimizers, Evaluators.
*   `deepfense/utils`: Registry and helpers.
*   `configs/`: Example YAML configurations.

## Usage

### Training

To start training, use `train.py` with a configuration file:

```bash
python train.py --config deepfense/config/your_config.yaml
```

### Configuration

The configuration file is divided into three main sections: `data`, `model`, and `training`.

```yaml
exp_name: "MyExperiment"
output_dir: "./outputs/"
seed: 42

data:
  sampling_rate: 16000
  train: ...
  val: ...

model:
  type: "StandardDetector"
  frontend: 
    type: "wav2vec2"
    args: { ckpt_path: "..." }
  backend: 
    type: "AASIST"
    args: { ... }
  loss:
    type: "AMSoftmax"
    embedding_dim: 160
    n_classes: 2
    m: 0.35
    s: 30

training:
  trainer: "StandardTrainer"
  epochs: 50
  optimizer: { type: "adam", lr: 0.0001 }
  device: "cuda"
```

See `docs/config.md` for detailed configuration options.

## Extending the Framework

See `docs/adding_components.md` for guides on adding:
*   New Models (Frontends/Backends)
*   New Losses
*   New Metrics

## License

[License Information]
