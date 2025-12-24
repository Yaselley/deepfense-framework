import pandas as pd
import argparse
from pathlib import Path

def process_one_csv(csv_path, data_root, exp_type, output_dir):
    """
    Output a parquet file for one csv file.
    For each item in the parquet files, it has four columns: ID, path, label, dataset_name.
    - ID: audio id, start from 0 and increment by 1 for each item.
    - path: absolute path to the audio file.
    - label: bonafide or spoof.
    - dataset_name: SpeechFake.
    """
    
    csv_name = csv_path.stem
    
    df_meta = pd.read_csv(csv_path)
    
    if df_meta.empty:
        print(f"Warning: CSV file is empty: {csv_path}, skipping")
        return
    
    if df_meta.columns[0].startswith('Unnamed'):
        df_meta = df_meta.drop(columns=[df_meta.columns[0]])
    
    if 'file' not in df_meta.columns:
        print(f"Warning: 'file' column not found in {csv_path}, skipping")
        return
    
    data_list = []
    audio_id = 0
    
    for _, row in df_meta.iterrows():
        relative_path = row['file']
        
        audio_path = data_root / relative_path
        
        #if not audio_path.exists():
        #    print(f"Warning: Audio file not found: {audio_path}")
        #    continue
        
        data_list.append({
            "ID": audio_id,
            "path": str(audio_path),
            "label": row['label'],
            "dataset_name": "SpeechFake"
        })
        audio_id += 1
    
    if data_list:
        exp_output_dir = output_dir / exp_type
        exp_output_dir.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(data_list)
        output_path = exp_output_dir / f"{csv_name}.parquet"
        df.to_parquet(output_path)
        print(f"Processed and saved {len(df)} entries from {exp_type}/{csv_name}.csv to {output_path}")
        print(f"Label distribution:")
        print(df['label'].value_counts())
        print()
    else:
        print(f"Warning: No valid entries found in {csv_path}, skipping")

def process_speechfake_dataset(data_root, meta_root, output_dir):
    """
    Process SpeechFake dataset by reading metadata CSV files from experiments directories.
    
    The dataset structure:
    - data_root/ contains BD/, MD/, Real/ directories with audio files
    - meta_root/metadata/experiments/ contains 4 experiment types:
      - baseline/
      - cross_generator/
      - cross_lingual/
      - cross_speaker/
    
    Each experiment type contains multiple CSV files that will be converted to parquet files.
    
    Args:
        data_root: root directory containing the processed dataset
                  Audio files should be in data_root/BD/, data_root/MD/, data_root/Real/
        meta_root: root directory containing metadata files
                  Metadata CSV files should be in meta_root/metadata/experiments/
        output_dir: directory where parquet files will be written
    """
    data_root = Path(data_root)
    meta_root = Path(meta_root)
    output_dir = Path(output_dir)
    
    experiments_dir = meta_root / "metadata" / "experiments"
    
    if not experiments_dir.exists():
        print(f"Error: Experiments directory not found: {experiments_dir}")
        return
    
    experiment_types = ["baseline", "cross_generator", "cross_lingual", "cross_speaker"]
    
    for exp_type in experiment_types:
        exp_dir = experiments_dir / exp_type
        
        if not exp_dir.exists():
            print(f"Warning: Experiment directory not found: {exp_dir}, skipping")
            continue
        
        csv_files = list(exp_dir.glob("*.csv"))
        
        if not csv_files:
            print(f"Warning: No CSV files found in {exp_dir}, skipping")
            continue
        
        for csv_path in csv_files:
            process_one_csv(csv_path, data_root, exp_type, output_dir)

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
    
    process_speechfake_dataset(data_root, meta_root, output_dir)

if __name__ == "__main__":
    main()

