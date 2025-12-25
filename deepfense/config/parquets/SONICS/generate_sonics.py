import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm

def process_single_split(meta_file, data_root, output_dir, real_songs_downloaded=False):
    """
    Process a single split of SONICS dataset and save directly to parquet file.
    
    Args:
        meta_file: Path to the metadata CSV file (e.g., train.csv, valid.csv, test.csv)
        data_root: Directory containing the processed audio files
        output_dir: Directory where to save the parquet file
        real_songs_downloaded: Whether real songs are downloaded (default: False)
    
    Returns:
        Number of rows processed, or 0 if error
    """
    split_name = meta_file.stem
    output_path = output_dir / f"{split_name}.parquet"

    if not meta_file.exists():
        print(f"Error: Metadata file not found: {meta_file}")
        return
    
    if not data_root.exists():
        print(f"Error: Data root directory not found: {data_root}")
        return

    all_data = []
    df = pd.read_csv(meta_file, low_memory=False)
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split_name} split"):
        filepath = row.get('filepath')
        filepath = str(filepath).strip()
        assert len(filepath) > 0, f"Filepath is empty for row {idx}"
        label = str(row.get('label')).strip()
        if not real_songs_downloaded and label == 'real':
            continue
        
        audio_path = data_root / filepath # e.g., fake_songs/fake_54113_suno_0.mp3
        
        if not audio_path.exists():
            print(f"Warning: Audio file not found: {audio_path}")
            continue
        file_id = str(row.get('id')).strip()
        
        all_data.append({
            "ID": file_id,
            "path": str(audio_path.relative_to(data_root)),
            "label": str(label),
            "dataset_name": "SONICS"
        })
    
    if all_data:
        df_output = pd.DataFrame(all_data)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_output.to_parquet(output_path)
        print(f"Saved {len(df_output)} {split_name} rows to {output_path}")
        print(f"{split_name} label distribution:")
        print(df_output['label'].value_counts())
    else:
        print(f"No data found for {split_name} split!")

def process_sonics_dataset(data_root, meta_root, output_dir, real_songs_downloaded=False):
    """
    Process SONICS dataset by reading metadata CSV files and scanning audio files.
    The dataset contains:
    - train.csv, valid.csv, test.csv (combined splits with real and fake songs)
    - fake_songs.csv, real_songs.csv (separate metadata files)
    - Labels: [real, full fake, half fake, mostly fake]
    - Audio files in data_root (relative paths in CSV start from data_root, e.g., fake_songs/fake_54113_suno_0.mp3)

    Directory structure:
        /mount/arbeitsdaten54/projekte/deepfake/fad/data/sonics/
        ├── processed/          (data_root)
        │   └── fake_songs/
        │       └── fake_*.mp3
        └── raw/
            └── sonics/         (meta_root)
                ├── README.md
                ├── metadata.json
                ├── train.csv
                ├── valid.csv
                ├── test.csv
                ├── fake_songs.csv
                ├── real_songs.csv
                └── fake_songs/

    Args:
        data_root: Directory containing the processed audio files (relative paths start from here)
        meta_root: Directory containing the metadata CSV files
        output_dir: Directory to save the parquet files
        real_songs_downloaded: Whether real songs are downloaded (default: False)
    """
    data_root = Path(data_root)
    meta_root = Path(meta_root)
    output_dir = Path(output_dir)
    
    if not meta_root.exists():
        print(f"Error: Metadata directory not found: {meta_root}")
        return
    
    if not data_root.exists():
        print(f"Error: Data root directory not found: {data_root}")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    split_names = ["train.csv", "valid.csv", "test.csv"]
    for file_name in split_names:
        meta_file = meta_root / file_name
        if not meta_file.exists():
            print(f"Error: Metadata file not found: {meta_file}")
            continue
        
        process_single_split(meta_file, data_root, output_dir, real_songs_downloaded)

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--meta_root", type=str, default=None)
    parser.add_argument("--real_songs_downloaded", action="store_true", default=False)

    output_dir = Path(__file__).parent.absolute()
    parser.add_argument("--output_dir", type=str, default=str(output_dir))
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    print(f"Writing parquet files to {output_dir}")
    
    data_root = Path(args.data_root)
    
    if args.meta_root is not None:
        meta_root = Path(args.meta_root)
    else:
        meta_root = data_root
    
    if not data_root.exists():
        print(f"Error: Data root directory not found: {data_root}")
        return
    
    process_sonics_dataset(data_root, meta_root, output_dir, args.real_songs_downloaded)

if __name__ == "__main__":
    main()

