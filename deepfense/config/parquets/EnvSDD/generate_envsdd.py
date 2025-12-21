import pandas as pd
import argparse
from pathlib import Path

def process_envsdd_split(data_root, split_name):
    """
    Process envsdd split by scanning all audio files.
    The dataset has:
    - {split_name}/fake_audio/: spoof (contains ATA and TTA subdirs with various models)
    - {split_name}/real_audio/: bonafide (only for development split)

    There are three splits in the original dataset: development, remain, and test.
    We only process the development and remain splits because the test split has no labels.
    Note: The same filename can appear in multiple locations (different models, splits, or fake/real),
    so we use relative path from data_root as the unique id.
    
    Args:
        data_root: root directory containing the processed envsdd dataset
        split_name: name of the split (e.g., "development", "remain")
    
    Returns:
        List of dicts containing id, path, label, and dataset_name
    """
    data_root = Path(data_root)
    if not data_root.exists():
        print(f"Error: Data root directory not found: {data_root}")
        return []
    
    split_data = []
    
    # Process fake_audio (spoof)
    fake_dir = data_root / split_name / "fake_audio"
    if fake_dir.exists():
        for wav_file in fake_dir.rglob("*.wav"):
            relative_path = wav_file.relative_to(data_root)
            audio_id = str(relative_path.with_suffix(""))
            
            split_data.append({
                "ID": audio_id,
                "path": str(wav_file),
                "label": "spoof",
                "dataset_name": "EnvSDD"
            })
    
    # Process real_audio (bonafide) - only for development split
    if split_name == "development":
        real_dir = data_root / split_name / "real_audio"
        if real_dir.exists():
            for wav_file in real_dir.rglob("*.wav"):
                relative_path = wav_file.relative_to(data_root)
                audio_id = str(relative_path.with_suffix(""))
                
                split_data.append({
                    "ID": audio_id,
                    "path": str(wav_file),
                    "label": "bonafide",
                    "dataset_name": "EnvSDD"
                })
    
    return split_data

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
    print(f"Writing parquet files to {output_dir}") 
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data_root = Path(args.data_root)
    meta_root = Path(args.meta_root) if args.meta_root is not None else data_root
    
    splits = ["development", "remain"]
    
    for split_name in splits:
        print(f"\nProcessing {split_name} split...")
        split_data = process_envsdd_split(data_root, split_name)
        if split_data:
            df = pd.DataFrame(split_data)
            output_path = output_dir / f"envsdd_{split_name}.parquet"
            df.to_parquet(output_path)
            print(f"Saved {len(df)} rows to {output_path}")
            print(f"Label distribution:")
            print(df['label'].value_counts())
        else:
            print(f"No {split_name} data to save!")

if __name__ == "__main__":
    main()

