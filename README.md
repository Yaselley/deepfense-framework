# DeepFense: Deepfake Detection Framework

**DeepFense** is a research framework for training and evaluating deepfake detection models.  
It supports flexible configuration, pretrained SSL model integration, and multilingual datasets.

---

## 🧩 Environment Setup

### 1. Create a Conda Environment
```bash
conda create -n deepfense python=3.10
conda activate deepfense
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Downgrade pip
After installation, make sure to downgrade pip to version 24.0:
```bash
pip install "pip<=24.0"
```

### 4. Build Fairseq

DeepFense relies on Fairseq for feature extraction and SSL model integration.
Navigate to the modules directory:

```bash
cd deepfense/models/
mkdir modules
cd modules
git clone https://github.com/facebookresearch/fairseq
cd fairseq
git checkout 3d262bb
pip install --editable ./
pip install --upgrade pip
```

### 5. Download Pretrained Weights
To obtain XLSR or other pretrained SSL model weights:

```bash
cd deepfense/models/pretrained
bash get_ckpts.sh
```

### 6. Prepare Your Dataset

Create a parquet file containing the audio samples and their labels.
Example format:

```python
{
  "path": "/path/to/audio.wav",
  "label": "bonafide"  # or "spoof"
}
```

Ensure that all audio files are accessible and correctly referenced in the parquet file.

### 7. Configure Training
Edit the training configuration file to match your dataset and experiment setup:

```bash
deepfense/config/train.yaml
```

### 8. Launch Training
Once your environment and configuration are ready, start training:

```bash
python3 train.py --config config/train.yaml
```
All logs, checkpoints, and configuration snapshots will be saved in the designated output directory.

