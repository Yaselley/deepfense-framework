import pandas as pd
import argparse
from pathlib import Path

def process_sonics_dataset(meta_file, audio_dir, real_songs_downloaded=False):
    """
    Process SONICS dataset by reading metadata CSV files and scanning audio files.
    The dataset contains:
    - train.csv, valid.csv, test.csv (combined splits with real and fake songs)
    - fake_songs.csv, real_songs.csv (separate metadata files)
    - Labels: [real, full fake, half fake, mostly fake]
    - Audio files in processed/ subdirectory (fake_songs/ and optionally real_songs/)

    Args:
        meta_file: Path to the metadata CSV file (e.g., train.csv, valid.csv, test.csv)
        audio_dir: Directory containing the processed audio files
        real_songs_downloaded: Whether real songs are downloaded (default: False)
    
    Returns:
        List of dicts containing ID, path, label, and dataset_name
    """
    meta_file = Path(meta_file)
    audio_dir = Path(audio_dir)

    if not meta_file.exists():
        print(f"Error: Metadata file not found: {meta_file}")
        return []
    
    if not audio_dir.exists():
        print(f"Error: Audio directory not found: {audio_dir}")
        return []

    all_data = []
    processed_files = set()
    
    try:
        df = pd.read_csv(meta_file, low_memory=False)
    except Exception as e:
        print(f"Error reading CSV file {meta_file}: {e}")
        return []
    
    if 'filepath' not in df.columns:
        print(f"Error: 'filepath' column not found in {meta_file}")
        return []
    
    for idx, row in df.iterrows():
        filepath = row.get('filepath', '')
        if pd.isna(filepath) or not filepath:
            continue
        
        filepath = str(filepath).strip()
        
        label = row.get('label', '')
        if not real_songs_downloaded and label == 'real':
            continue
        
        audio_path = audio_dir / filepath # e.g., fake_songs/fake_54113_suno_0.mp3
        
        if not audio_path.exists():
            print(f"Warning: Audio file not found: {audio_path}")
            continue
        
        if filepath in processed_files:
            continue
        processed_files.add(filepath)
        
        filename = row.get('filename', '')
        file_id = row.get('id', '')
        
        if filename and pd.notna(filename):
            file_id = Path(filename).stem
        elif file_id and pd.notna(file_id):
            file_id = str(file_id)
        else:
            file_id = Path(filepath).stem
        
        label = row.get('label', '')
        if pd.isna(label):
            label = ''
        
        all_data.append({
            "ID": file_id,
            "path": str(audio_path),
            "label": str(label),
            "dataset_name": "SONICS"
        })
    
    return all_data

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
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data_root = Path(args.data_root)
    
    if args.meta_root is not None:
        meta_root = Path(args.meta_root)
    else:
        meta_root = data_root
    
    if not data_root.exists():
        print(f"Error: Data root directory not found: {data_root}")
        return
    
    if not meta_root.exists():
        print(f"Error: Metadata directory not found: {meta_root}")
        return
    
    splits = [
        ("train.csv", "train"),
        ("valid.csv", "valid"),
        ("test.csv", "test")
    ]
    
    for meta_filename, split_name in splits:
        meta_file = meta_root / meta_filename
        print(f"\nProcessing {split_name} split...")
        all_data = process_sonics_dataset(meta_file, data_root, args.real_songs_downloaded)
        
        if all_data:
            df = pd.DataFrame(all_data)
            output_path = output_dir / f"{split_name}.parquet"
            df.to_parquet(output_path)
            print(f"Saved {len(df)} {split_name} rows to {output_path}")
            print(f"{split_name} label distribution:")
            print(df['label'].value_counts())
        else:
            print(f"No data found for {split_name} split!")

if __name__ == "__main__":
    main()

