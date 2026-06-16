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

Audio is **not** pre-windowed at 20 ms in the dataloader. The same **full clip** tensor is fed to the SSL frontend; the frontend’s convolutional stride defines the temporal resolution (≈20 ms per frame for common 16 kHz Wav2Vec2 setups with total stride 320). Dense labels in the parquet must be **aligned to that resolution** (or cropped/padded; see below).

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

Required:

- `path` — audio path (same as `StandardDataset`).

One of:

- `frame_labels` — list/array of class indices (length = number of frames for the clip at the chosen resolution), or comma-separated string, or JSON list string.
- `frame_labels_path` — path to a `.npy` / `.npz` (1D int array).

Optional:

- `label` — clip-level class; if omitted, a **weak** label is set to spoof iff any frame equals `label_map[\"spoof\"]`.

Data config (train/val) should include:

- `dataset_type: TemporalSegmentationDataset`
- `label_hop` or `label_hop_ms` — prediction rate (default 320 samples ≈ 20 ms @ 16 kHz)
- `source_label_hop` or `source_label_hop_ms` — annotation rate if labels are stored at a finer resolution (defaults to `label_hop`)
- `label_merge_rule` — `any_spoof` (default), `all_spoof`, or `majority` when downsampling labels
- `max_frames` — optional cap on frame labels per clip

Clips are always returned at full length; variable-length batches are padded by `collate_fn`. Invalid tail frames are labeled `-100` and ignored by the loss.

## Example model config (_yaml excerpt_)

```yaml
model:
  type: TemporalDetector
  label_hop_ms: 40
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
