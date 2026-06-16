# Temporal / partial deepfake detection in DeepFense

This guide covers **dense per-frame (per time-step) labels** for partial deepfake and PartialSpoof: `TemporalSegmentationDataset`, `TemporalDetector`, framewise losses, and localization metrics (`RANGE_EER`, `SEGMENT_EER`, `MULTIRES_EER`).

## Quick start

```bash
git fetch origin
git checkout deepfense-partial
pip install -e .
deepfense train --config deepfense/config/experiments/temporal_deepfake_example.yaml
```

Copy `deepfense/config/experiments/temporal_deepfake_example.yaml` and set parquet paths, checkpoint paths, and `output_dir`. Ready-made PartialSpoof configs live under `deepfense/config/experiments/PartialSpoof/`.

**ReadTheDocs:** clip-level docs stay on [`/en/latest/`](https://deepfense.readthedocs.io/en/latest/). After activating the `deepfense-partial` branch as a ReadTheDocs version, partial docs build at `/en/deepfense-partial/` (or your chosen slug).

## Design summary

| Area | Clip-level (original) | Temporal (new) |
|------|------------------------|----------------|
| Dataset | `StandardDataset`, parquet `path` + `label` | `TemporalSegmentationDataset`, `path` + `frame_labels` (and optional `frame_labels_path`, `label`) |
| Waveform | Fixed-length padding; mask `(B, T_audio)` | Same; plus frame targets aligned to approximate SSL frame rate |
| Model | `StandardDetector` → backend pools to `(B, D)` | `TemporalDetector` → `FrameMLP` keeps `(B, T_frames, D)` |
| Loss | `CrossEntropy` on `(B, C)` | `FramewiseCrossEntropy` on `(B, T, C)` with `ignore_index=-100` |
| Val metrics | EER / ACC / F1 on one score per clip | `FRAME_*`, `SEGMENT_EER`, `RANGE_EER`, `MULTIRES_EER` |

Audio is **not** pre-windowed at 20 ms in the dataloader. The same **full clip** tensor is fed to the SSL frontend; **`model.frontend_hop`** (samples per SSL frame) must match your frontend — **320** for Wav2Vec2 / WavLM / HuBERT @ 16 kHz. Dense labels in the parquet are at **`source_label_hop_ms`**; the model predicts at **`label_hop_ms`**.

## Timing hops (`frontend_hop`, `label_hop`, `source_label_hop`)

Three related rates — **set `frontend_hop` explicitly in YAML**:

| Key | Where | Meaning | Example @ 16 kHz |
|-----|--------|---------|------------------|
| **`model.frontend_hop`** | `model:` | SSL **native** frame stride (samples). One Wav2Vec2 frame every `frontend_hop` samples. Used to pool SSL features → `label_hop` and to map batch **audio mask → frame mask**. | **`320`** (20 ms) |
| `data.label_hop_ms` / `label_hop` | `data:` → copied to `model` by `train.py` | **Model output** / loss frame rate. Must be an **integer multiple** of `frontend_hop`. | `40` ms → 640 samples → `pool_factor = 2` |
| `data.source_label_hop_ms` | `data:` (dataset only) | **Parquet annotation** rate. PartialSpoof labels are usually 20 ms. | `20` ms → 320 samples |

**Rules**

1. `label_hop % frontend_hop == 0` (else `TemporalDetector` raises at init).
2. `label_hop % source_label_hop == 0` (else dataset raises at init).
3. Always set **`model.frontend_hop: 320`** for standard SSL frontends (do not rely on silent defaults).

```yaml
model:
  type: TemporalDetector
  frontend_hop: 320          # REQUIRED — SSL stride; masking + pooling depend on this
  pool_mode: mean            # when label_hop > frontend_hop
  # label_hop_ms copied from data: by train.py
```

## Batch padding and masks

Variable-length clips are **zero-padded in `collate_fn`**, not fixed-padded in the dataset.

| Tensor | Shape | Values | Role |
|--------|--------|--------|------|
| `mask` | `(B, T_audio)` | 1 = real audio, 0 = batch pad | Passed to SSL frontend + downsampled to frames in `TemporalDetector` |
| `frame_labels` | `(B, T_frames)` | class id or **`-100`** | `FramewiseCrossEntropy` ignores `-100` |
| `frame_mask` | `(B, T_frames)` | 1 = valid, 0 = batch pad | Used at **eval** metrics only |

Inside `TemporalDetector.forward(x, mask)`:

1. `frontend(x, mask)` — SSL ignores padded audio samples.
2. Pool SSL features from `frontend_hop` → `label_hop` if needed (`pool_mode`).
3. `downsample_mask_to_frames(mask, label_hop)` → zero invalid frame embeddings.
4. Loss uses `frame_labels`; positions with `-100` are skipped.

See `deepfense/models/temporal_detector.py` and `deepfense/data/data_utils.py` (`collate_fn`).

## Files added or changed

### Added

- `deepfense/data/temporal_utils.py` — crop/pad label vectors; frame count from audio length and hop; label downsampling merge rules.
- `deepfense/data/temporal_dataset.py` — `TemporalSegmentationDataset`; full-audio clips with `-100` on padded/invalid tail frames.
- `deepfense/models/backends/frame_mlp.py` — `FrameMLP`: MLP-style projection without pooling.
- `deepfense/models/temporal_detector.py` — `TemporalDetector`: `frontend → frame backend → framewise loss head`.
- `deepfense/models/losses/framewise_ce.py` — `FramewiseCrossEntropy`: masked CE; logits `(B, T, C)`; aligns `T` if logits and labels differ slightly.
- `docs/temporal_deepfake.md` — this document.

### Changed

- `deepfense/data/data_utils.py` — `collate_fn` pads optional `frame_labels`, `frame_mask`; supports `(N_aug, T)` waveform batches; imports `deepfense.data` so dataset registrations load.
- `deepfense/data/__init__.py` — imports temporal dataset module.
- `deepfense/training/standard_trainer.py` — training/eval uses `frame_labels` when present; concat-aug repeats framewise targets; validation flattens framewise LLR scores for metrics.
- `deepfense/training/evaluations/evaluator.py` — imports `deepfense.training.evaluations` so all built-in metrics register reliably.
- `deepfense/training/evaluations/metrics.py` — `FRAME_*` metrics.
- `deepfense/cli/commands/test.py` — temporal evaluation path; skips legacy per-utterance text export when framewise.
- `deepfense/models/__init__.py`, `backends/__init__.py`, `losses/__init__.py` — imports to register new modules.

## Parquet schema (`TemporalSegmentationDataset`)

Each row is one utterance. Labels are **integer class indices** (same values as `label_map`, e.g. `0` = spoof, `1` = bonafide). One label per frame at **`source_label_hop_ms`** (not at `label_hop_ms` — the dataloader downsamples if those differ).

### Required columns

| Column | Description |
|--------|-------------|
| `path` | Audio file path (same as clip-level `StandardDataset`). Relative paths are joined with `root_dir` if set. |

### Frame labels — pick **one** of these columns

| Column | When to use |
|--------|-------------|
| `frame_labels` | Store labels **inline** in the parquet (short/medium clips). |
| `frame_labels_path` | Store labels in an external **`.npy` / `.npz`** file (long clips, large datasets). |

If **both** are present on a row, **`frame_labels_path` wins** (inline `frame_labels` is ignored for that row).

### `frame_labels` — supported cell formats

The loader accepts several parquet cell types. All parse to a 1D `int` vector (one value per annotated frame).

| Format | Parquet / pandas type | Example cell | Notes |
|--------|----------------------|--------------|-------|
| **List (recommended)** | `list` / Arrow `list<int>` | `[1, 1, 0, 0, 1, 1]` | Best when building parquets with PyArrow/pandas. |
| **NumPy array** | `array` | `array([1, 1, 0, 0])` | Accepted after `read_parquet`. |
| **Comma-separated string** | `string` | `"1,1,0,0,1,1"` | Spaces optional: `"1, 1, 0, 0"`. |
| **JSON list string** | `string` | `"[1, 1, 0, 0, 1, 1]"` | Must start with `[` and end with `]`. |

**Do not** put file paths in `frame_labels` — paths ending in `.npy` / `.npz` raise an error. Use the `frame_labels_path` column instead.

**Minimal pandas example (inline list column):**

```python
import pandas as pd

df = pd.DataFrame({
    "ID": ["utt_001"],
    "path": ["/data/audio/utt_001.wav"],
    "frame_labels": [[1, 1, 0, 0, 1, 1]],  # bonafide=1, spoof=0 @ source_label_hop
})
df.to_parquet("train.parquet")
```

**Minimal pandas example (comma-separated string):**

```python
df = pd.DataFrame({
    "path": ["/data/audio/utt_001.wav"],
    "frame_labels": ["1,1,0,0,1,1"],
})
```

### `frame_labels_path` — external label files

| Field | Format |
|-------|--------|
| Cell value | Path to a **1D** `.npy` file, or `.npz` (first array in the archive is used). |
| Path | Absolute, or relative to `root_dir` (same rules as audio `path`). |
| Contents | 1D integer array, length = number of frames at `source_label_hop_ms`. |

```python
import numpy as np
import pandas as pd

np.save("/data/labels/utt_001.npy", np.array([1, 1, 0, 0, 1, 1], dtype=np.int64))

df = pd.DataFrame({
    "path": ["/data/audio/utt_001.wav"],
    "frame_labels_path": ["/data/labels/utt_001.npy"],
})
```

### Optional columns

| Column | Description |
|--------|-------------|
| `label` | Clip-level class (`"bonafide"` / `"spoof"` strings mapped via `label_map`). If omitted, a **weak** clip label is inferred: spoof if **any** frame equals `label_map["spoof"]`, else bonafide. |
| `ID` | Utterance id (recommended for exports and `MULTIRES_EER`). |

### Length and timing

- Label length should match the audio duration at **`source_label_hop_ms`** (e.g. PartialSpoof @ 20 ms → `source_label_hop_ms: 20`).
- If length differs by more than 1 frame, a warning is logged and labels are aligned (crop/pad with ignore index `-100` on invalid tail frames).
- If `label_hop_ms` > `source_label_hop_ms`, fine labels are merged using `label_merge_rule` (`any_spoof`, `all_spoof`, `majority`, `any_non_bonafide`).

### Data config (train/val)

- `dataset_type: TemporalSegmentationDataset`
- `label_hop` or `label_hop_ms` — model prediction rate (default 320 samples ≈ 20 ms @ 16 kHz)
- `source_label_hop` or `source_label_hop_ms` — parquet annotation rate (defaults to `label_hop`)
- `label_merge_rule` — merge rule when downsampling labels
- `max_frames` — optional cap on frame labels per clip after crop

Clips are returned at full length (or cropped with `max_len`); variable-length batches are padded by `collate_fn`. Invalid tail frames use label `-100` and are ignored by the loss.

## Example model config (_yaml excerpt_)

```yaml
model:
  type: TemporalDetector
  frontend_hop: 320                  # SSL frame stride (samples); required for masking
  label_hop_ms: 40                   # or set under data: — copied by train.py
  pool_mode: mean
  frontend:
    type: wav2vec2
    args:
      source: fairseq
      ckpt_path: /path/to/xlsr2_300m.pt
      freeze: false
  backend:
    type: FrameMLP
    args:
      input_dim: 1024
      projection: [512]
      activation: relu
      norm_type: layer
  # Alternative temporal backend:
  # backend:
  #   type: GMLP
  #   args:
  #     input_dim: 1024
  #     d_ffn: -2
  #     seq_len: 512
  #     gmlp_layers: 5
  #     pooling: none
  #     output_dim: 512
  loss:
    - type: FramewiseCrossEntropy
      weight: 1.0
      embedding_dim: 512
      n_classes: 2
      ignore_index: -100

training:
  monitor_metric: RANGE_EER_20ms
  monitor_mode: min
  metrics:
    FRAME_ACC: {}
    FRAME_F1: { f1_average: macro }
    FRAME_AUC: {}
    FRAME_JACCARD_SPOOF: { spoof_label: 0 }
    SEGMENT_EER: {}
    RANGE_EER:
      label_hop_ms: 20
    MULTIRES_EER:
      resolutions_ms: [20, 40, 80, 160]
```

Use `embedding_dim` equal to the **last** dimension after `FrameMLP` or `GMLP` (with `projection: [512]` or `output_dim: 512` that is `512`; with empty `projection` it is `input_dim`).

## Evaluation metrics

Partial spoof evaluation has three tiers (all **lower is better** for EER-style metrics → `monitor_mode: min`):

| Tier | Metrics | Purpose |
|------|---------|---------|
| 1 — Framewise | `FRAME_ACC`, `FRAME_F1`, `FRAME_AUC`, `FRAME_JACCARD_SPOOF` | Per-frame classification quality |
| 2 — Native resolution | `SEGMENT_EER`, `RANGE_EER` | PartialSpoof protocol metrics at training hop |
| 3 — Multi-resolution | `MULTIRES_EER` | `SEGMENT_EER` + `RANGE_EER` at 20/40/80/160 ms; also `UTTERANCE_EER` |

`MULTIRES_EER` logs keys such as `RANGE_EER_20ms`, `SEGMENT_EER_40ms`, and concatenated percent strings `RANGE_EER_CONCAT_pct`. Common choices for `monitor_metric`: `RANGE_EER`, `RANGE_EER_20ms`, or `MULTIRES_EER`.

## Relation to wedefense

Legacy wedefense localization helpers (RTTM IO, reference range-EER scripts) remain under `wedefense/wedefense/metrics/localization/` in this branch if you need to compare against older tooling.

## Limitations / next steps

- **Exact frame count**: HuggingFace Wav2Vec2 feature lengths can differ slightly from `floor(n_samples / 320)`. `FramewiseCrossEntropy` pads or crops labels in time to match logits; for production, consider reading `output_lengths` from the frontend or matching wedefense’s resolution helpers.
- **Secondary losses** (e.g. `AMSoftmax`) are not wired for `(B, T, D)`; use `FramewiseCrossEntropy` as the primary loss until margin losses are extended.
- **EER** on framewise scores is supported mathematically but may not match community “range EER” protocols for partial spoof; use `FRAME_AUC` / segment metrics or port wedefense’s range-EER.
