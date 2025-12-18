import pandas as pd
import argparse
from pathlib import Path

# Hardcoded label map: only 'original' maps to 'bonafide'
LABEL_MAP = {
    'original': 'bonafide'
}

def process_compspoof_dataset(meta_file, audio_dir):
    """
    Process CompSpoof dataset by reading metadata files and scanning audio files.
    The dataset contains:
    - CompSpoof_train.txt, CompSpoof_dev.txt, CompSpoof_eval.txt
    - Format: mixed_audio speech_source env_source class_label (4 space-separated fields)
    - Each line has three audio files: mixed_audio, speech_source, env_source
    - Labels are mapped using LABEL_MAP, with 'original' mapping to 'bonafide'

    Args:
        meta_file: Path to the metadata text file (e.g., CompSpoof_train.txt)
        audio_dir: Directory containing the audio files
    
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
    audio_id_set = set()
    
    with open(meta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # mixed_audio speech_source env_source class_label
            parts = line.split()
            
            mixed_audio = parts[0]
            speech_source = parts[1]
            env_source = parts[2]
            class_label = parts[3]
            
            mixed_label = LABEL_MAP.get(class_label, class_label)
            
            mixed_audio_path = audio_dir / mixed_audio
            if mixed_audio_path.exists():
                if mixed_audio not in audio_id_set:
                    audio_id_set.add(mixed_audio)
                    file_id = Path(mixed_audio).stem
                    all_data.append({
                        "ID": file_id,
                        "path": str(mixed_audio_path),
                        "label": mixed_label,
                        "dataset_name": "CompSpoof"
                    })
            else:
                print(f"Warning: Mixed audio file not found: {mixed_audio_path}")

            if class_label == 'original':
                continue
            
            label_parts = class_label.split('_')
            speech_label, env_label = label_parts
            
            speech_source_path = audio_dir / speech_source
            if speech_source_path.exists():
                if speech_source not in audio_id_set:
                    audio_id_set.add(speech_source)
                    file_id = Path(speech_source).stem
                    all_data.append({
                        "ID": file_id,
                        "path": str(speech_source_path),
                        "label": speech_label,
                        "dataset_name": "CompSpoof"
                    })
            else:
                print(f"Warning: Speech source file not found: {speech_source_path}")
            
            env_source_path = audio_dir / env_source
            if env_source_path.exists():
                if env_source not in audio_id_set:
                    audio_id_set.add(env_source)
                    file_id = Path(env_source).stem
                    all_data.append({
                        "ID": file_id,
                        "path": str(env_source_path),
                        "label": env_label,
                        "dataset_name": "CompSpoof"
                    })
            else:
                print(f"Warning: Env source file not found: {env_source_path}")
    
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
        required=None,
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
    if meta_root is None:
        meta_root = data_root
        print(f"Meta directory not provided, using data directory as meta directory")
    meta_root = Path(args.meta_root)
    
    splits = [
        ("CompSpoof_train.txt", "train"),
        ("CompSpoof_dev.txt", "dev"),
        ("CompSpoof_eval.txt", "eval")
    ]
    
    for meta_filename, split_name in splits:
        meta_file = meta_root / meta_filename
        all_data = process_compspoof_dataset(meta_file, data_root)
        
        if all_data:
            df = pd.DataFrame(all_data)
            output_path = output_dir / f"compspoof_{split_name}.parquet"
            df.to_parquet(output_path)
            print(f"\nSaved {len(df)} {split_name} rows to {output_path}")
            print(f"{split_name} label distribution:")
            print(df['label'].value_counts())
        else:
            print(f"No data found for {split_name} split!")

if __name__ == "__main__":
    main()
