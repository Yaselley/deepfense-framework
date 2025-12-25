# Installation Guide for DeepFense

## Quick Install (PyPI)

Once published to PyPI, users can install DeepFense with:

```bash
pip install deepfense
```

## Development Install

For development or if installing from source:

```bash
# Clone the repository
git clone https://github.com/Yaselley/deepfense-framework
cd DeepFense

# Install in editable mode
pip install -e .
```

## Post-Installation: Fairseq (Optional)

If you plan to use fairseq-based frontends (Wav2Vec2, HuBERT via fairseq), you need to install Fairseq separately:

```bash
# Downgrade pip (Required for Fairseq)
pip install "pip<=24.0"

# Install Fairseq
mkdir -p deepfense/models/modules
cd deepfense/models/modules
git clone https://github.com/facebookresearch/fairseq
cd fairseq
git checkout 3d262bb
pip install --editable ./

# Optional: Upgrade pip back if needed
# pip install --upgrade pip
cd ../../../..
```

**Note**: If you only use HuggingFace-based frontends, you can skip this step.

## Verify Installation

```bash
python -c "import deepfense; print(deepfense.__version__)"
```

## Troubleshooting

### Import Errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- For fairseq frontends, ensure Fairseq is installed separately

### CUDA Issues
- Install PyTorch with CUDA support: https://pytorch.org/get-started/locally/

### Missing Config Files
- Config files are included in the package. If missing, reinstall: `pip install --force-reinstall deepfense`
