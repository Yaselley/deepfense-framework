import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm

def process_spoofceleb_dataset(data_root, meta_root, output_dir):
    """
    Process SpoofCeleb dataset by reading metadata CSV files.
    The dataset contains:
    - a00/ dir: bonafide (genuine human speech)
    - a01/ through a23/ dirs: spoof (TTS-generated speech)
    https://jungjee.github.io/spoofceleb/
    SpoofCeleb: A Large-Scale Public Audio-Visual Dataset for Deepfake Detection
    
    Data structure tree for data_root (/mount/arbeitsdaten54/projekte/deepfake/fad/data/spoofceleb/processed):
    data_root/
    └── spoofceleb/
        ├── flac/
        │   ├── train/
        │   │   ├── a00/ (bonafide)
        │   │   │   ├── speaker subdirectories (e.g., id10310/)
        │   │   │   └── *.flac (audio files)
        │   │   └── a01/ through a10/ (spoof)
        │   │       ├── speaker subdir
        │   │       └── *.flac (audio files)
        │   ├── development/
        │   │   ├── a00/ (bonafide)
        │   │   │   ├── speaker subdir
        │   │   │   └── *.flac (audio files)
        │   │   └── a06/, a07/, a11/, a12/, a13/, a14/ (spoof)
        │   │       ├── speaker subdir
        │   │       └── *.flac (audio files)
        │   └── evaluation/
        │       ├── a00/ (bonafide)
        │       │   ├── speaker subdir
        │       │   └── *.flac (audio files)
        │       └── a15/ through a23/ (spoof)
        │           ├── speaker subdir
        │           └── *.flac (audio files)
        └── metadata/
            ├── train.csv
            ├── development.csv
            └── evaluation.csv
    
    Args:
        data_root: root directory containing the processed dataset 
                  Audio files should be put in data_root / spoofceleb / flac / ...
        meta_root: root directory containing metadata files
                  Metadata csv files should be put in meta_root / spoofceleb / metadata/*.csv
        output_dir: directory where parquet files will be written
    """
    data_root = Path(data_root)
    meta_root = Path(meta_root)
    output_dir = Path(output_dir)
    
    flac_dir = data_root / "spoofceleb"/ "flac"
    
    if not flac_dir.exists():
        print(f"Error: FLAC directory not found: {flac_dir}")
        return
    
    metadata_dir = meta_root / "spoofceleb" / "metadata"
    
    if not metadata_dir.exists():
        print(f"Error: Metadata directory not found: {metadata_dir}")
        return
    
    csv_files = list(metadata_dir.glob("*.csv"))
    
    if not csv_files:
        print(f"Warning: No CSV files found in {metadata_dir}")
        return
    
    for csv_path in csv_files:
        split_name = csv_path.stem
        
        split_data = []
        df_meta = pd.read_csv(csv_path)
        print(f"Processing {split_name} split with {len(df_meta)} rows")
        
        for _, row in tqdm(df_meta.iterrows(), total=len(df_meta), desc=f"Processing {split_name} split"):
            relative_path = row['file']
            attack_type = row['attack']
            
            audio_path = flac_dir / split_name / relative_path
            
            #if not audio_path.exists():
            #    print(f"Warning: Audio file not found: {audio_path}")
            #    continue
            
            audio_id = audio_path.stem
            
            if attack_type == "a00":
                label = "bonafide"
            else:
                label = "spoof"
            
            split_data.append({
                "ID": audio_id,
                "path": str(audio_path.relative_to(data_root)),
                "label": label,
                "dataset_name": "SpoofCeleb"
            })
        
        if split_data:
            df = pd.DataFrame(split_data)
            output_path = output_dir / f"{split_name}.parquet"
            df.to_parquet(output_path)
            print(f"\nSaved {len(df)} {split_name} rows to {output_path}")
            print(f"{split_name} label distribution:")
            print(df['label'].value_counts())
        else:
            print(f"No data found for {split_name} split!")

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
    process_spoofceleb_dataset(data_root, meta_root, output_dir)

if __name__ == "__main__":
    main()
