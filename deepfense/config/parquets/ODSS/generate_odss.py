import pandas as pd
import argparse
from pathlib import Path

def process_odss_dataset(data_root):
    """
    Process ODSS dataset by scanning all audio files.
    The dataset contains:
    - natural/ dir: bonafide 
    - fastpitch-hifigan/, vits/ dirs: spoof 
    https://www.idmt.fraunhofer.de/en/publications/datasets/ODSS-open-dataset-synthetic-speech.html
    An Open Dataset of Synthetic Speech, a multilingual, multispeaker dataset of synthetic and natural speech.

    Data structure tree for data_root (/mount/arbeitsdaten54/projekte/deepfake/fad/data/odss/processed):
    data_root/
    └── odss/
        ├── natural/ (bonafide)
        │   ├── hifi-tts/
        │   │   ├── speaker subdirectories (e.g., 92/)
        │   │   └── *.wav (audio files in each speaker subdirectory)
        │   ├── hui-acg/
        │   │   ├── speaker subdirectories (e.g., Bernd_Ungerer/)
        │   │   └── *.wav (audio files in each speaker subdirectory)
        │   ├── openslr-es/
        │   │   ├── speaker subdirectories (e.g., arf_00295/)
        │   │   └── *.wav (audio files in each speaker subdirectory)
        │   └── vctk/
        │       ├── speaker subdirectories (e.g., p227/)
        │       └── *.wav (audio files in each speaker subdirectory)
        ├── fastpitch-hifigan/ (spoof)
        │   ├── hifi-tts/ (Same sub dirs as natural/hifi-tts)
        │   ├── hui-acg/ 
        │   └── openslr-es/
        └── vits/ (spoof)
            ├── hifi-tts/
            ├── hui-acg/
            ├── openslr-es/
            └── vctk/

    Args:
        data_root: root directory containing the uncompressed ODSS dataset (odss/ subdirectory will be appended)
    
    Returns:
        List of dicts containing ID, path, label, and dataset_name
    """
    data_root = Path(data_root)
    if not data_root.exists():
        print(f"Error: Data root directory not found: {data_root}")
        return []
    
    all_data = []
    
    natural_dir = data_root / "natural"
    if natural_dir.exists():
        for wav_file in natural_dir.rglob("*.wav"):
            audio_id = wav_file.stem
            
            all_data.append({
                "ID": audio_id,
                "path": str(wav_file.relative_to(data_root)),
                "label": "bonafide",
                "dataset_name": "ODSS"
            })
    
    fastpitch_dir = data_root / "fastpitch-hifigan"
    if fastpitch_dir.exists():
        for wav_file in fastpitch_dir.rglob("*.wav"):
            audio_id = wav_file.stem
            all_data.append({
                "ID": audio_id,
                "path": str(wav_file.relative_to(data_root)), 
                "label": "spoof",
                "dataset_name": "ODSS"
            })

    vits_dir = data_root / "vits"
    if vits_dir.exists():
        for wav_file in vits_dir.rglob("*.wav"):
            audio_id = wav_file.stem
            all_data.append({
                "ID": audio_id,
                "path": str(wav_file.relative_to(data_root)),
                "label": "spoof",
                "dataset_name": "ODSS"
            })
    
    return all_data

def main():
    parser = argparse.ArgumentParser(
        description="Generate parquet files for ODSS dataset. It requires the uncompressed dataset root as input, the default output dir is the same directory as this script.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
    )
    
    parser.add_argument(
        "--meta_root",
        type=str,
        default=None,
    )
    
    output_dir = Path(__file__).parent.absolute()
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(output_dir),
    )
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    print(f"Writing parquet files to {output_dir}") 
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data_root = Path(args.data_root)
    data_root = data_root / "odss"
    meta_root = Path(args.meta_root) if args.meta_root is not None else data_root
    all_data = process_odss_dataset(data_root)
    
    if all_data:
        df = pd.DataFrame(all_data)
        output_path = output_dir / "odss.parquet"
        df.to_parquet(output_path)
        print(f"\nSaved {len(df)} rows to {output_path}")
        print(f"Label distribution:")
        print(df['label'].value_counts())
        print(f"Should have 30025 rows, got {len(df)}")
    else:
        print("No data to save!")

if __name__ == "__main__":
    main()
