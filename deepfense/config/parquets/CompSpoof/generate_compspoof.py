import pandas as pd
import argparse
from pathlib import Path

def process_single_meta_file(meta_file, audio_dir, data_root, output_dir):
    """
    Process a single CompSpoof metadata file and save the parquet.
    
    Args:
        meta_file: Path to the metadata text file (e.g., CompSpoof_train.txt)
        audio_dir: Directory containing the audio files
        data_root: Root directory for computing relative paths
        output_dir: Directory to save the parquet file
    """
    if not meta_file.exists():
        print(f"Warning: Metadata file not found: {meta_file}")
        return
    
    all_data = []
    audio_id_set = set()
    
    with open(meta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # mixed_audio speech_source env_source mixed_label
            parts = line.split()
            
            mixed_audio = parts[0]
            # parts[1] is speech_source (not used)
            # parts[2] is env_source (not used)
            mixed_label = parts[3]
            
            mixed_audio_path = audio_dir / mixed_audio
            if mixed_audio_path.exists():
                if mixed_audio not in audio_id_set:
                    audio_id_set.add(mixed_audio)
                    file_id = Path(mixed_audio).stem
                    all_data.append({
                        "ID": file_id,
                        "path": str(mixed_audio_path.relative_to(data_root)),
                        "label": mixed_label,
                        "dataset_name": "CompSpoof"
                    })
            else:
                print(f"Warning: Mixed audio file not found: {mixed_audio_path}")
    
    if not all_data:
        print(f"No data found for {meta_file.name}!")
        return
    
    # Save the parquet file (name matches the txt file name without extension)
    df = pd.DataFrame(all_data)
    output_filename = meta_file.stem + ".parquet"
    output_path = output_dir / output_filename
    df.to_parquet(output_path)
    print(f"\nSaved {len(df)} rows to {output_path}")
    print(f"{meta_file.stem} label distribution:")
    print(df['label'].value_counts())

def process_compspoof_dataset(data_root, meta_root, output_dir):
    """
    Process CompSpoof dataset by reading metadata files and scanning audio files.
    The dataset contains:
    - CompSpoof_train.txt, CompSpoof_dev.txt, CompSpoof_eval.txt
    - Format: mixed_audio speech_source env_source class_label (4 space-separated fields)
    - Only the mixed_audio files are processed with their mixed_label
    - Labels: [original, bonafide_bonafide, spoof_bonafide, bonafide_spoof, spoof_spoof]

    Args:
        data_root: Root directory containing the audio files
            - If the data_root is the processed dir, then it has the following structure:
            data_root/
            └── dataset/
                └── (audio files)

        meta_root: Root directory containing the protocol txt files
            - Structure:
            meta_root/
            ├── CompSpoof_dev.txt
            ├── CompSpoof_eval.txt
            ├── CompSpoof_train.txt
            ├── LICENSE.md
            └── README.md

        output_dir: Directory to save the parquet files
    """
    data_root = Path(data_root)
    meta_root = Path(meta_root)
    output_dir = Path(output_dir)
    
    audio_dir = data_root / "dataset"
    
    if not audio_dir.exists():
        print(f"Error: Audio directory not found: {audio_dir}")
        return
    
    for meta_file in sorted(meta_root.glob("CompSpoof_*.txt")):
        process_single_meta_file(meta_file, audio_dir, data_root, output_dir)

def main():
    parser = argparse.ArgumentParser()
    
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
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data_root = Path(args.data_root)
    meta_root = Path(args.meta_root) if args.meta_root is not None else data_root
    
    process_compspoof_dataset(data_root, meta_root, output_dir)

if __name__ == "__main__":
    main()
