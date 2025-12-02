# Inference Guide

To run inference on a trained model (e.g., for submission to a leaderboard or testing).

## Using `test.py`

The `test.py` script loads a checkpoint and runs the model on a target dataset.

```bash
python test.py \
    --config config/train.yaml \
    --checkpoint outputs/MyExp/ckpts/best_model.pth \
    --output_file submission.txt
```

## Interactive Inference

You can also load the model in a notebook or script:

```python
import torch
from deepfense.utils.registry import build_detector
import yaml

# 1. Load Config
with open("config/train.yaml") as f:
    config = yaml.safe_load(f)

# 2. Build Model
model = build_detector(config["model"]["type"], config["model"])

# 3. Load Checkpoint
ckpt = torch.load("outputs/MyExp/ckpts/best_model.pth")
model.load_state_dict(ckpt["model_state"])
model.eval()

# 4. Run
audio = torch.randn(1, 64600) # Load your audio here
with torch.no_grad():
    output = model(audio)
    score = output["scores"]
    print(f"Spoof Score: {score.item()}")
```

