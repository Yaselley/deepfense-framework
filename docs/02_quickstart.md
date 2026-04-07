# Quick Start

Train your first model in 5 minutes.

---

## 1. Generate Test Data

DeepFense includes a helper to create dummy data for testing:

```bash
python tests/create_samples.py --output-dir ./tests
```

This creates `tests/test.parquet` with 20 samples pointing to a dummy WAV file.

---

## 2. Check the Config

The example config at `deepfense/config/train.yaml` is already set up to use this test data. Open it and check:

- **Frontend**: Wav2Vec2 from HuggingFace (downloads automatically)
- **Backend**: Nes2Net
- **Loss**: CrossEntropy + AMSoftmax
- **Data**: Points to `./tests/test.parquet`

---

## 3. Train

```bash
python train.py --config deepfense/config/train.yaml
```

You'll see:
```
[INFO] Experiment directory: outputs/Wav2Vec2_Nes2Net_Example_20250401_120000
[INFO] Trainable parameters: 1,234,567
Epoch 1/10: 100%|██████████| 2/2 [00:05<00:00]
[INFO] Average Metrics: loss: 0.6931, EER: 50.00
```

(EER = 50% is expected on random dummy data.)

---

## 4. Test

```bash
python test.py \
    --config deepfense/config/train.yaml \
    --checkpoint outputs/Wav2Vec2_Nes2Net_Example_*/best_model.pth
```

---

## 5. Use Your Own Data

Replace the dummy data with your real dataset:

```python
import pandas as pd

df = pd.DataFrame({
    "ID": ["utt_001", "utt_002", "utt_003"],
    "path": ["/data/audio1.flac", "/data/audio2.flac", "/data/audio3.flac"],
    "label": ["bonafide", "spoof", "bonafide"],
})
df.to_parquet("my_train.parquet")
```

Then update the config:

```yaml
data:
  train:
    parquet_files: ["my_train.parquet"]
  val:
    parquet_files: ["my_val.parquet"]
```

---

## Output Structure

After training, check `outputs/YOUR_EXP_NAME/`:

```
outputs/Wav2Vec2_Nes2Net_Example_20250401_120000/
├── config.yaml          # Saved config (for reproducibility)
├── train.log            # Full training log
├── best_model.pth       # Best checkpoint (by monitor_metric)
├── ckpts/               # All checkpoints
├── results/             # Metrics JSON files
└── plots/               # Training curves (loss, EER, etc.)
```

---

**Next:** [Full Tutorial](03_full_tutorial.md) -- complete walkthrough of every config option
