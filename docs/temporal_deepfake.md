# Temporal / partial deepfake detection in DeepFense

This note describes how clip-level DeepFense was extended for **dense per-frame (per time-step) labels**, inspired by the framewise localization loop in **wedefense** (flattened `CrossEntropy` on `B×T` logits) while reusing DeepFense’s SSL frontends, YAML/registry wiring, and training CLI.

## Design summary

| Area | Clip-level (original) | Temporal (new) |
|------|------------------------|----------------|
| Dataset | `StandardDataset`, parquet `path` + `label` | `TemporalSegmentationDataset`, `path` + `frame_labels` (and optional `frame_labels_path`, `label`) |
| Waveform | Fixed-length padding; mask `(B, T_audio)` | Same; plus frame targets aligned to approximate SSL frame rate |
| Model | `StandardDetector` → backend pools to `(B, D)` | `TemporalDetector` → `FrameMLP` keeps `(B, T_frames, D)` |
| Loss | `CrossEntropy` on `(B, C)` | `FramewiseCrossEntropy` on `(B, T, C)` with `ignore_index=-100` |
| Val metrics | EER / ACC / F1 on one score per clip | `FRAME_ACC`, `FRAME_F1`, `FRAME_AUC`, `FRAME_JACCARD_SPOOF` on flattened valid frames |

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
  loss:
    - type: FramewiseCrossEntropy
      weight: 1.0
      embedding_dim: 512
      n_classes: 2
      ignore_index: -100

training:
  monitor_metric: FRAME_F1
  monitor_mode: max
  metrics:
    FRAME_ACC: {}
    FRAME_F1: { f1_average: macro }
    FRAME_AUC: {}
    FRAME_JACCARD_SPOOF: { spoof_label: 0 }
```

Use `embedding_dim` equal to the **last** dimension after `FrameMLP` (with `projection: [512]` that is `512`; with empty `projection` it is `input_dim`).

## Relation to wedefense

For rich partial-spoof tooling (range-EER, RTTM IO), see `/netscratch/yelkheir/DeepFense/DeepFense/wedefense/wedefense/metrics/localization/` and `/netscratch/yelkheir/DeepFense/DeepFense/wedefense/wedefense/dataset/processor_time.py`.

## Limitations / next steps

- **Exact frame count**: HuggingFace Wav2Vec2 feature lengths can differ slightly from `floor(n_samples / 320)`. `FramewiseCrossEntropy` pads or crops labels in time to match logits; for production, consider reading `output_lengths` from the frontend or matching wedefense’s resolution helpers.
- **Secondary losses** (e.g. `AMSoftmax`) are not wired for `(B, T, D)`; use `FramewiseCrossEntropy` as the primary loss until margin losses are extended.
- **EER** on framewise scores is supported mathematically but may not match community “range EER” protocols for partial spoof; use `FRAME_AUC` / segment metrics or port wedefense’s range-EER.
