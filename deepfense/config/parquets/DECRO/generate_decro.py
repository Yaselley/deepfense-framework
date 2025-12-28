import pandas as pd
import argparse
from pathlib import Path

def process_decro_protocol(data_root, meta_root, output_dir):
    """
    Process decro protocol files (format: ID ID - - METHOD Label)
    The dataest is used to evaluate how language differences affect deepfake detection. It includes two languages: English and Chinese.
    
    Args:
        data_root: Root dir containing the audio files (The processed dir by AUDDIT)
            - If the data_root is the processed dir by AUDDIT, then it has the following structure:
            data_root/
            └── petrichorwq-DECRO-dataset-6fc9884/
                ├── ch_dev/          # Audio files directory
                ├── ch_dev.txt        # Protocol file
                ├── ch_eval/
                ├── ch_eval.txt
                ├── ch_train/
                ├── ch_train.txt
                ├── en_dev/
                ├── en_dev.txt
                ├── en_eval/
                ├── en_eval.txt
                ├── en_train/
                ├── en_train.txt
                ├── LICENSE
                └── README.md

        meta_root: Root dir containing the protocol txt files
        output_dir: Directory to save the parquet files
    """
    data_root = Path(data_root)
    meta_root = Path(meta_root)
    output_dir = Path(output_dir)
    
    audio_base_dir = data_root / "petrichorwq-DECRO-dataset-6fc9884"
    
    for txt_file in sorted(meta_root.glob("*.txt")):
        split_name = txt_file.stem
        
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
                
                audio_dir = audio_base_dir / split_name
                file_path = audio_dir / f"{audio_id}.wav"
                
                data.append({
                    "ID": audio_id,
                    "path": str(file_path.relative_to(data_root)),
                    "label": label,
                    "dataset_name": "DECRO"
                })
        
        if not data:
            print(f"No data found for {split_name}")
            continue
        
        df = pd.DataFrame(data)
        output_path = output_dir / f"{split_name}.parquet"
        df.to_parquet(output_path)
        print(f"\nSaved {len(df)} {split_name} rows to {output_path}")
        print(f"{split_name} label distribution:")
        print(df['label'].value_counts())

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
    
    parser.add_argument(
        "--meta_root",
        type=str,
        default=None,
    )

    script_dir = Path(__file__).parent.absolute()
    output_dir = script_dir
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(output_dir),
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Keep original data_root for relative paths
    data_root = Path(args.data_root)
    meta_root = Path(args.meta_root) if args.meta_root is not None else data_root / "petrichorwq-DECRO-dataset-6fc9884"
    
    process_decro_protocol(data_root, meta_root, output_dir)

if __name__ == "__main__":
    main()
