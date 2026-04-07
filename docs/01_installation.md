# Installation

## Prerequisites

- **Python** 3.10+
- **PyTorch** 2.1+ (with CUDA recommended)
- **OS**: Linux, macOS, or Windows

## Install from Source (Recommended)

```bash
# 1. Clone
git clone https://github.com/Yaselley/deepfense-framework
cd deepfense-framework

# 2. Create environment
conda create -n deepfense python=3.10
conda activate deepfense

# 3. Install
pip install -e .
```

This installs DeepFense in editable mode with all dependencies (torch, transformers, wandb, etc.).

## Verify

```bash
python -c "import deepfense; print('OK')"
deepfense list   # shows all registered components
```

## Frontend Checkpoints

DeepFense supports two ways to load pretrained SSL models:

| Method | Config | What you need |
|--------|--------|---------------|
| **HuggingFace** | `source: "huggingface"` | Just an internet connection (downloads automatically) |
| **Local .pt file** | `source: "fairseq"` | A downloaded `.pt` checkpoint file |

**No separate fairseq installation is needed.** DeepFense includes a built-in loader for fairseq-format `.pt` checkpoints.

Example -- HuggingFace (easiest):
```yaml
frontend:
  type: "wav2vec2"
  args:
    source: "huggingface"
    ckpt_path: "facebook/wav2vec2-xls-r-300m"
```

Example -- local `.pt` file:
```yaml
frontend:
  type: "wav2vec2"
  args:
    source: "fairseq"
    ckpt_path: "/path/to/xlsr2_300m.pt"
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: torch` | Install PyTorch first: https://pytorch.org/get-started/locally/ |
| `CUDA not available` | `python -c "import torch; print(torch.cuda.is_available())"` -- install CUDA-enabled PyTorch |
| `deepfense: command not found` | Run `pip install -e .` again from the repo root |

---

**Next:** [Quick Start](02_quickstart.md)
