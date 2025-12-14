import pandas as pd
import os

# Configuration
ASV5_TRAIN_TXT = "/netscratch/yelkheir/datasets/asv5/asvspoof5.train.txt"
ASV5_DEV_42_TXT = "/netscratch/yelkheir/datasets/asv5/asvspoof5.dev.42.txt"  # Dev split with seed 42
ASV5_DEV_FULL_TXT = "/netscratch/yelkheir/datasets/asv5/asvspoof5.dev.txt"  # Full dev split
ASV5_EVAL_TXT = "/netscratch/yelkheir/datasets/asv5/asvspoof5.eval.txt"
OUTPUT_DIR = "/netscratch/yelkheir/DeepFense/DeepFense/deepfense/config/parquets/ASV5"

# Root directories for audio files
ASV5_TRAIN_ROOT = "/ds-slt/audio/ASVSpoof2024/flac_T"
ASV5_DEV_ROOT = "/ds-slt/audio/ASVSpoof2024/flac_D"
ASV5_EVAL_ROOT = "/ds-slt/audio/ASVSpoof2024/Eval/flac_E_eval"

def process_protocol(txt_file, root_dir, split_name, dataset_name="ASVSpoof5"):
    """Process ASV5 protocol file (train, dev, or eval)"""
    print(f"Processing {split_name} data from {txt_file}...")
    
    data = []
    with open(txt_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            
            # Format for train/dev: SPEAKER_ID AUDIO_ID - ATTACK_TYPE LABEL
            # Example: T_4850 T_0000000000 - A05 spoof
            # Format for eval: SPEAKER_ID AUDIO_ID - - LABEL
            # Example: E_1607 E_0009538969 - - spoof
            
            audio_id = parts[1]  # Column 1 (0-indexed): Audio ID
            label = parts[-1]    # Label is last column
            
            # Sanity check on label
            if label not in ["bonafide", "spoof"]:
                continue
            
            # Audio files are named as {audio_id}.flac
            file_path = os.path.join(root_dir, f"{audio_id}.flac")
            
            data.append({
                "ID": audio_id,
                "path": file_path,
                "label": label,
                "dataset_name": dataset_name
            })
    
    print(f"Processed {len(data)} {split_name} entries")
    return data

def main():
    # Process train split
    train_data = process_protocol(
        ASV5_TRAIN_TXT,
        ASV5_TRAIN_ROOT,
        "train"
    )
    
    # Process dev.42 (validation) split
    dev_42_data = process_protocol(
        ASV5_DEV_42_TXT,
        ASV5_DEV_ROOT,
        "dev.42"
    )
    
    # Process full dev (validation) split
    dev_full_data = process_protocol(
        ASV5_DEV_FULL_TXT,
        ASV5_DEV_ROOT,
        "dev.full"
    )
    
    # Process eval (test) split
    eval_data = process_protocol(
        ASV5_EVAL_TXT,
        ASV5_EVAL_ROOT,
        "eval"
    )
    
    # Save train to parquet file
    if train_data:
        train_df = pd.DataFrame(train_data)
        train_output_path = os.path.join(OUTPUT_DIR, "asvspoof5_train.parquet")
        train_df.to_parquet(train_output_path)
        print(f"\nSaved {len(train_df)} train rows to {train_output_path}")
        print(f"Train label distribution:")
        print(train_df['label'].value_counts())
    
    # Save dev.42 to parquet file
    if dev_42_data:
        dev_42_df = pd.DataFrame(dev_42_data)
        dev_42_output_path = os.path.join(OUTPUT_DIR, "asvspoof5_val.parquet")
        dev_42_df.to_parquet(dev_42_output_path)
        print(f"\nSaved {len(dev_42_df)} dev.42 rows to {dev_42_output_path}")
        print(f"Dev.42 label distribution:")
        print(dev_42_df['label'].value_counts())
    
    # Save full dev to parquet file
    if dev_full_data:
        dev_full_df = pd.DataFrame(dev_full_data)
        dev_full_output_path = os.path.join(OUTPUT_DIR, "asvspoof5_val_full.parquet")
        dev_full_df.to_parquet(dev_full_output_path)
        print(f"\nSaved {len(dev_full_df)} dev.full rows to {dev_full_output_path}")
        print(f"Dev.full label distribution:")
        print(dev_full_df['label'].value_counts())
    
    # Save eval to parquet file
    if eval_data:
        eval_df = pd.DataFrame(eval_data)
        eval_output_path = os.path.join(OUTPUT_DIR, "asvspoof5_test.parquet")
        eval_df.to_parquet(eval_output_path)
        print(f"\nSaved {len(eval_df)} eval rows to {eval_output_path}")
        print(f"Eval label distribution:")
        print(eval_df['label'].value_counts())

if __name__ == "__main__":
    main()
