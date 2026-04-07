"""Generate sample parquet files for testing DeepFense."""
import argparse
import numpy as np
import pandas as pd
import soundfile as sf
import tempfile
import os


def create_dummy_wav(path, duration_sec=1.0, sr=16000):
    """Create a short silent WAV file for testing."""
    samples = np.zeros(int(sr * duration_sec), dtype=np.float32)
    sf.write(path, samples, sr)


def main():
    parser = argparse.ArgumentParser(description="Generate sample test data for DeepFense")
    parser.add_argument("--output-dir", type=str, default="./tests", help="Directory to save parquet files")
    parser.add_argument("--n-samples", type=int, default=20, help="Number of samples per split")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    wav_dir = os.path.join(args.output_dir, "dummy_audio")
    os.makedirs(wav_dir, exist_ok=True)

    wav_path = os.path.join(wav_dir, "test.wav")
    create_dummy_wav(wav_path)

    n = args.n_samples
    df = pd.DataFrame({
        "ID": [f"sample_{i:04d}" for i in range(n)],
        "path": [wav_path] * n,
        "label": np.random.choice(["bonafide", "spoof"], size=n),
        "dataset_name": ["test_dataset"] * n,
    })

    df.to_parquet(os.path.join(args.output_dir, "test.parquet"), index=False)
    print(f"Created {args.output_dir}/test.parquet with {n} samples")


if __name__ == "__main__":
    main()
