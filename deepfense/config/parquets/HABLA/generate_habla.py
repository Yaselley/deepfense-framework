import pandas as pd
import os

# Configuration
HABLA_TRAIN_TXT = "/netscratch/yelkheir/datasets/spanish/train_out.txt"
HABLA_DEV_TXT = "/netscratch/yelkheir/datasets/spanish/val_out.txt"
HABLA_EVAL_TXT = "/netscratch/yelkheir/datasets/spanish/test_out.txt"
OUTPUT_DIR = "/netscratch/yelkheir/DeepFense/DeepFense/deepfense/config/parquets/HABLA"

# Root directory for audio files (same for all splits)
HABLA_AUDIO_ROOT = "/ds/audio/latina_america/wav"

def process_habla_protocol(txt_file, root_dir, split_name, dataset_name="HABLA"):
    """Process HABLA protocol file (format: ID ID _ _ LABEL)"""
    print(f"Processing {split_name} data from {txt_file}...")
    
    data = []
    with open(txt_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            
            # Format: ID ID _ _ LABEL
            # Example: arf_00295_00001008290 arf_00295_00001008290 _ _ bonafide
            audio_id = parts[1]  # Second column is the audio ID
            label = parts[-1]    # Label is last column
            
            # Sanity check on label
            if label not in ["bonafide", "spoof"]:
                continue
            
            # Audio files are named as {audio_id}.wav
            file_path = os.path.join(root_dir, f"{audio_id}.wav")
            
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
    train_data = process_habla_protocol(
        HABLA_TRAIN_TXT,
        HABLA_AUDIO_ROOT,
        "train"
    )
    
    # Process dev (validation) split
    dev_data = process_habla_protocol(
        HABLA_DEV_TXT,
        HABLA_AUDIO_ROOT,
        "dev"
    )
    
    # Process eval (test) split
    eval_data = process_habla_protocol(
        HABLA_EVAL_TXT,
        HABLA_AUDIO_ROOT,
        "eval"
    )
    
    # Save train to parquet file
    if train_data:
        train_df = pd.DataFrame(train_data)
        train_output_path = os.path.join(OUTPUT_DIR, "habla_train.parquet")
        train_df.to_parquet(train_output_path)
        print(f"\nSaved {len(train_df)} train rows to {train_output_path}")
        print(f"Train label distribution:")
        print(train_df['label'].value_counts())
    
    # Save dev to parquet file
    if dev_data:
        dev_df = pd.DataFrame(dev_data)
        dev_output_path = os.path.join(OUTPUT_DIR, "habla_val.parquet")
        dev_df.to_parquet(dev_output_path)
        print(f"\nSaved {len(dev_df)} dev rows to {dev_output_path}")
        print(f"Dev label distribution:")
        print(dev_df['label'].value_counts())
    
    # Save eval to parquet file
    if eval_data:
        eval_df = pd.DataFrame(eval_data)
        eval_output_path = os.path.join(OUTPUT_DIR, "habla_test.parquet")
        eval_df.to_parquet(eval_output_path)
        print(f"\nSaved {len(eval_df)} eval rows to {eval_output_path}")
        print(f"Eval label distribution:")
        print(eval_df['label'].value_counts())

if __name__ == "__main__":
    main()
