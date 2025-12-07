# Getting Started

This guide will help you set up DeepFense and run your first training experiment.

## Prerequisites

*   Python 3.8+
*   PyTorch 1.10+
*   Torchaudio
*   Pandas, Numpy, Scipy
*   SoundFile, Librosa

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/DeepFense.git
    cd DeepFense
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Data Preparation

DeepFense expects data metadata in **Parquet** format. The Parquet file should contain at least:
*   `path`: Absolute path to the audio file.
*   `label`: String label (e.g., "bonafide", "spoof").
*   `dataset_name`: (Optional) Source dataset name (e.g., "ASVSpoof19").

Example creating a dummy parquet file:
```python
import pandas as pd
df = pd.DataFrame({
    "path": ["/path/to/audio1.flac", "/path/to/audio2.flac"],
    "label": ["bonafide", "spoof"]
})
df.to_parquet("train.parquet")
```

## Running Training

1.  **Configure `config/train.yaml`**:
    Update the `parquet_files` paths to point to your data.
    ```yaml
    data:
      train:
        parquet_files: ["/abs/path/to/train.parquet"]
      val:
        parquet_files: ["/abs/path/to/val.parquet"]
    ```

2.  **Start Training**:
    ```bash
    python train.py --config config/train.yaml
    ```

    This will:
    *   Load the config.
    *   Initialize the `StandardDetector` with the defined Frontend and Backend.
    *   Start the `StandardTrainer`.
    *   Save outputs to `outputs/EXP_NAME/`.

## Monitoring

*   **Console**: Progress bars and metrics are printed to stdout.
*   **WandB**: If enabled in config, logs are sent to Weights & Biases.
*   **Plots**: EER/Loss trend plots are saved in `outputs/.../plots/`.
