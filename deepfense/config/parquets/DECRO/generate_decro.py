import pandas as pd
import os
import argparse
from pathlib import Path

def process_decro_protocol(txt_file, root_dir, split_name):
    """
    Process decro protocol files (format: ID ID - - METHOD Label)
    The dataest is used to evaluate how language differences affect deepfake detection. It includes two languages: English and Chinese.
    
    Args:
        txt_file: Path to the protocol text file
        root_dir: Root dir containing the audio files (The processed dir by AUDDIT)
        split_name: Nmae of the split (e..g, 'en_train', 'ch_dev')
    
    Returns:
        List of dict containing ID, path, label, and dataset_name
    """
    txt_file = Path(txt_file)
    root_dir = Path(root_dir)
    if not txt_file.exists():
        print(f"Warning: Protocol file not found: {txt_file}")
        return []
    
    data = []
    with open(txt_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            
            # SPEAKER_ID AUDIO_FILE_NAME - SYSTEM_ID KEY
            # Example: 4 4-727-124443-0055 - baidu spoof
            assert len(parts) == 5, f"Expected 5 parts, got {len(parts)}"
            
            audio_id = parts[1]  
            label = parts[-1]    
            
            # Audio files are in subdirectories named after the split (e.g., en_train/, ch_dev/)
            # Extract language and split type from split_name (e.g., 'en_train' -> 'en_train')
            audio_dir = root_dir / split_name
            file_path = audio_dir / f"{audio_id}.wav"
            
            data.append({
                "ID": audio_id,
                "path": str(file_path),
                "label": label,
                "dataset_name": "DECRO"
            })
    
    print(f"Processed {len(data)} {split_name} entries")
    return data

def main():
    parser = argparse.ArgumentParser(
        description="Generate parquet files for DECRO dataset, it requires the uncompressed dataset root as input, the default output dir is the same directory as this script.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory containing the processed DECRO dataset"
    )

    script_dir = Path(__file__).parent.absolute()
    output_dir = script_dir / "DECRO"
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(output_dir),
    )
    
    args = parser.parse_args()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data_root = Path(args.data_root)
    splits = [
        ("en_train.txt", "en_train"),
        ("en_dev.txt", "en_dev"),
        ("en_eval.txt", "en_eval"),
        ("ch_train.txt", "ch_train"),
        ("ch_dev.txt", "ch_dev"),
        ("ch_eval.txt", "ch_eval")
    ]
    
    all_data = {}
    for protocol_file, split_name in splits:
        txt_file = data_root / protocol_file
        split_data = process_decro_protocol(txt_file, data_root, split_name)
        all_data[split_name] = split_data
    
    for split_name in ["en_train", "ch_train"]:
        if all_data.get(split_name):
            df = pd.DataFrame(all_data[split_name])
            output_path = output_dir / f"decro_{split_name}.parquet"
            df.to_parquet(output_path)
            print(f"\nSaved {len(df)} {split_name} rows to {output_path}")
            print(f"{split_name} label distribution:")
            print(df['label'].value_counts())
    
    for split_name in ["en_dev", "ch_dev"]:
        if all_data.get(split_name):
            df = pd.DataFrame(all_data[split_name])
            output_path = output_dir / f"decro_{split_name}.parquet"
            df.to_parquet(output_path)
            print(f"\nSaved {len(df)} {split_name} rows to {output_path}")
            print(f"{split_name} label distribution:")
            print(df['label'].value_counts())
    
    for split_name in ["en_eval", "ch_eval"]:
        if all_data.get(split_name):
            df = pd.DataFrame(all_data[split_name])
            output_path = output_dir / f"decro_{split_name}.parquet"
            df.to_parquet(output_path)
            print(f"\nSaved {len(df)} {split_name} rows to {output_path}")
            print(f"{split_name} label distribution:")
            print(df['label'].value_counts())

if __name__ == "__main__":
    main()
