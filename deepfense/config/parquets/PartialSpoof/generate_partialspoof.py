#!/usr/bin/env python3
"""Generate PartialSpoof parquets with 20 ms segment labels.

Uses the standard (non-adjusted) protocol files and the official PartialSpoof
segment label npy files at 0.02 s hop (20 ms @ 16 kHz → 320 samples).

Output schema (TemporalSegmentationDataset):
  - path          : absolute path to wav
  - ID            : utterance id (e.g. CON_T_0000029)
  - label         : clip-level bonafide | spoof
  - speaker       : speaker id
  - system_id     : spoof system id (or '-')
  - frame_labels  : list[int] per-frame labels (0=spoof, 1=bonafide) @ 20 ms
  - dataset_name  : PartialSpoof
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Defaults
PROTOCOL_DIR = "/netscratch/yelkheir/datasets/partial_asvspoof"
AUDIO_ROOT = "/ds-slt/audio/PartialSpoof/database"
SEGLAB_DIR = os.path.join(AUDIO_ROOT, "segment_labels")
OUTPUT_DIR = Path(__file__).resolve().parent

# 0.02 s = 20 ms hop (320 samples @ 16 kHz)
SOURCE_LABEL_HOP_SEC = 0.02

SPLITS = {
    "train": "train.txt",
    "dev": "dev.txt",
    "eval": "eval.txt",
}


def load_segment_labels(split: str, seglab_dir: str) -> dict[str, np.ndarray]:
    seglab_path = os.path.join(
        seglab_dir, f"{split}_seglab_{SOURCE_LABEL_HOP_SEC:.2f}.npy"
    )
    if not os.path.isfile(seglab_path):
        raise FileNotFoundError(f"Segment label file not found: {seglab_path}")
    return np.load(seglab_path, allow_pickle=True).item()


def parse_protocol_line(line: str) -> tuple[str, str, str, str] | None:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    speaker_id, audio_id, _, system_id, clip_label = parts[:5]
    if clip_label not in {"bonafide", "spoof"}:
        return None
    return speaker_id, audio_id, system_id, clip_label


def frame_labels_for_utt(seglab: dict, audio_id: str) -> list[int]:
    if audio_id not in seglab:
        raise KeyError(f"Missing segment labels for {audio_id}")
    raw = seglab[audio_id]
    # Official labels are stored as '0' (spoof) / '1' (bonafide) strings.
    return [int(x) for x in raw.reshape(-1)]


def process_split(
    split: str,
    protocol_file: str,
    seglab: dict,
    protocol_dir: str,
    audio_root: str,
) -> pd.DataFrame:
    protocol_path = os.path.join(protocol_dir, protocol_file)
    if not os.path.isfile(protocol_path):
        raise FileNotFoundError(f"Protocol file not found: {protocol_path}")

    wav_dir = os.path.join(audio_root, split, "con_wav")
    rows: list[dict] = []
    missing_labels = 0
    missing_audio = 0

    with open(protocol_path, "r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_protocol_line(line)
            if parsed is None:
                continue
            speaker_id, audio_id, system_id, clip_label = parsed

            wav_path = os.path.join(wav_dir, f"{audio_id}.wav")
            if not os.path.isfile(wav_path):
                missing_audio += 1
                continue

            try:
                frame_labels = frame_labels_for_utt(seglab, audio_id)
            except KeyError:
                missing_labels += 1
                continue

            rows.append(
                {
                    "ID": audio_id,
                    "path": wav_path,
                    "label": clip_label,
                    "speaker": speaker_id,
                    "system_id": system_id,
                    "frame_labels": frame_labels,
                    "dataset_name": "PartialSpoof",
                }
            )

    if missing_audio:
        print(f"  [{split}] skipped {missing_audio} rows (missing wav)")
    if missing_labels:
        print(f"  [{split}] skipped {missing_labels} rows (missing segment labels)")

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PartialSpoof parquets")
    parser.add_argument(
        "--protocol-dir",
        default=PROTOCOL_DIR,
        help="Directory with train.txt / dev.txt / eval.txt",
    )
    parser.add_argument(
        "--audio-root",
        default=AUDIO_ROOT,
        help="PartialSpoof database root (contains train/dev/eval/con_wav)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Where to write *.parquet files",
    )
    args = parser.parse_args()

    protocol_dir = args.protocol_dir
    audio_root = args.audio_root
    seglab_dir = os.path.join(audio_root, "segment_labels")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Protocol dir : {protocol_dir}")
    print(f"Audio root   : {audio_root}")
    print(f"Segment labs : {seglab_dir} (*_seglab_{SOURCE_LABEL_HOP_SEC:.2f}.npy)")
    print(f"Output dir   : {output_dir}")
    print(f"Source hop   : {SOURCE_LABEL_HOP_SEC * 1000:.0f} ms")

    for split, protocol_file in SPLITS.items():
        print(f"\nProcessing {split} ...")
        seglab = load_segment_labels(split, seglab_dir)
        df = process_split(split, protocol_file, seglab, protocol_dir, audio_root)

        out_path = output_dir / f"{split}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"  Saved {len(df)} rows -> {out_path}")

        n_frames = df["frame_labels"].map(len)
        print(
            f"  frame_labels: min={n_frames.min()}, max={n_frames.max()}, "
            f"mean={n_frames.mean():.1f}"
        )
        print(
            f"  clip labels: bonafide={(df['label'] == 'bonafide').sum()}, "
            f"spoof={(df['label'] == 'spoof').sum()}"
        )


if __name__ == "__main__":
    main()
