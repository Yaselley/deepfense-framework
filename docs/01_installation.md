# Installation Guide

This guide covers all installation methods for DeepFense.

---

## Prerequisites

<<<<<<< HEAD
- **Python**: 3.8 or higher
- **PyTorch**: 2.0+ (with CUDA support recommended)
=======
- **Python**: 3.10 or higher (3.10 recommended)
>>>>>>> ceeccd5 (Switch to ReadTheDocs theme)
- **OS**: Linux, macOS, or Windows

---

## Option 1: Install from PyPI (Recommended)

```bash
pip install deepfense
```

---

## Option 2: Install from Source (Development)

### Step 1: Clone the Repository

```bash
git clone https://github.com/Yaselley/deepfense-framework
cd deepfense-framework
```

### Step 2: Create Virtual Environment (Recommended)

Using **Conda**:
```bash
conda create -n deepfense python=3.10
conda activate deepfense
```

Or using **venv**:
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows
```

### Step 3: Install Core Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install DeepFense in Editable Mode

```bash
pip install -e .
```

---

## Optional: Fairseq Installation

> **Note**: Only required if you plan to use **Fairseq-based frontends** (Wav2Vec2, HuBERT via Fairseq checkpoints). Skip this if you only use HuggingFace models.

Fairseq requires an older pip version to build correctly:

```bash
# Downgrade pip temporarily
pip install "pip<=24.0"

# Clone and install Fairseq
mkdir -p deepfense/models/modules
cd deepfense/models/modules
git clone https://github.com/facebookresearch/fairseq
cd fairseq
git checkout 3d262bb
pip install --editable ./

# Return to project root
cd ../../../..

# Optional: Upgrade pip back
pip install --upgrade pip
```

---

## Verify Installation

```bash
python -c "import deepfense; print('DeepFense installed successfully!')"
```

---

## Troubleshooting

### Import Errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- For Fairseq frontends, verify Fairseq is installed separately

### CUDA Issues
- Install PyTorch with CUDA support from: https://pytorch.org/get-started/locally/
- Verify CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

### Missing Config Files
- Config files are included in the package
- If missing, reinstall: `pip install --force-reinstall deepfense`

---

> **Next Step**: [Quick Start →](02_quickstart.md)
