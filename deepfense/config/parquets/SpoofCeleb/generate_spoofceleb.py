import pandas as pd
import argparse
from pathlib import Path

def process_spoofceleb_dataset(data_root, meta_root):
    """
    Process SpoofCeleb dataset by reading metadata CSV files.
    The dataset contains:
    - a00/ dir: bonafide (genuine human speech)
    - a01/ through a23/ dirs: spoof (TTS-generated speech)
    https://jungjee.github.io/spoofceleb/
    SpoofCeleb: A Large-Scale Public Audio-Visual Dataset for Deepfake Detection
    
    The dataset structure:
    - flac/train/, flac/development/, flac/evaluation/ contain audio files
    - metadata/ contains CSV files with file paths, speaker IDs, and attack types
    
    Args:
        data_root: root directory containing the processed dataset 
                  Audio files should be put in data_root / spoofceleb / flac / ...
        meta_root: root directory containing metadata files
                  Metadata csv files should be put in meta_root / spoofceleb / metadata/*.csv
    
    Returns:
        List of dicts containing ID, path, label, and dataset_name
    """
    data_root = Path(data_root)
    meta_root = Path(meta_root)
    
    flac_dir = data_root / "spoofceleb"/ "flac"
    
    if not flac_dir.exists():
        print(f"Error: FLAC directory not found: {flac_dir}")
        return []
    
    all_data = []
    
    splits = {
        "train": "train.csv",
        "development": "development.csv",
        "evaluation": "evaluation.csv"
    }
    
    metadata_dir = meta_root / "spoofceleb" / "metadata"
    
    if not metadata_dir.exists():
        print(f"Error: Metadata directory not found: {metadata_dir}")
        return []
    
    for split_name, csv_filename in splits.items():
        csv_path = metadata_dir / csv_filename
        if not csv_path.exists():
            print(f"Warning: Metadata file not found: {csv_path}, skipping split {split_name}")
            continue
        
        df_meta = pd.read_csv(csv_path)
        
        for _, row in df_meta.iterrows():
            relative_path = row['file']
            attack_type = row['attack']
            
            audio_path = flac_dir / split_name / relative_path
            
            if not audio_path.exists():
                print(f"Warning: Audio file not found: {audio_path}")
                continue
            
            audio_id = audio_path.stem
            
            if attack_type == "a00":
                label = "bonafide"
            else:
                label = "spoof"
            
            all_data.append({
                "ID": audio_id,
                "path": str(audio_path),
                "label": label,
                "dataset_name": "SpoofCeleb"
            })
    
    return all_data

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
    all_data = process_spoofceleb_dataset(data_root, meta_root)
    
    if all_data:
        df = pd.DataFrame(all_data)
        output_path = output_dir / "spoofceleb.parquet"
        df.to_parquet(output_path)
        print(f"\nSaved {len(df)} rows to {output_path}")
        print(f"Label distribution:")
        print(df['label'].value_counts())
    else:
        print("No data to save!")

if __name__ == "__main__":
    main()
