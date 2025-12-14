import pandas as pd
import os

# Configuration
ADD23_TRAIN_TXT = "/netscratch/yelkheir/datasets/add_2023/train.txt"
ADD23_DEV_TXT = "/netscratch/yelkheir/datasets/add_2023/dev.txt"
ADD23_EVAL_TXT = "/netscratch/yelkheir/datasets/add_2023/eval.txt"  # testR1
ADD23_EVAL2_TXT = "/netscratch/yelkheir/datasets/add_2023/eval.2.txt"  # testR2
OUTPUT_DIR = "/netscratch/yelkheir/DeepFense/DeepFense/deepfense/config/parquets/ADD23"

# Root directories for audio files
ADD23_TRAIN_ROOT = "/ds/audio/ADD23_track_1.2/Track1.2/train/wav"
ADD23_DEV_ROOT = "/ds/audio/ADD23_track_1.2/Track1.2/dev/wav"
ADD23_EVAL_R1_ROOT = "/ds/audio/ADD23_track_1.2/Track1.2/testR1/wav"
ADD23_EVAL_R2_ROOT = "/ds/audio/ADD23_track_1.2/Track1.2/testR2/wav"

def process_add23_protocol(txt_file, root_dir, split_name, dataset_name="ADD23"):
    """Process ADD23 protocol file (format: ID ID _ _ LABEL)"""
    print(f"Processing {split_name} data from {txt_file}...")
    
    data = []
    with open(txt_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            
            # Format: ID ID _ _ LABEL
            # Example: ADD2023_T1.2_T_00000000 ADD2023_T1.2_T_00000000 _ _ spoof
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
    train_data = process_add23_protocol(
        ADD23_TRAIN_TXT,
        ADD23_TRAIN_ROOT,
        "train"
    )
    
    # Process dev split
    dev_data = process_add23_protocol(
        ADD23_DEV_TXT,
        ADD23_DEV_ROOT,
        "dev"
    )
    
    # Process eval R1 (testR1)
    eval_r1_data = process_add23_protocol(
        ADD23_EVAL_TXT,
        ADD23_EVAL_R1_ROOT,
        "eval.R1"
    )
    
    # Process eval R2 (testR2)
    eval_r2_data = process_add23_protocol(
        ADD23_EVAL2_TXT,
        ADD23_EVAL_R2_ROOT,
        "eval.R2"
    )
    
    # Save train to parquet file
    if train_data:
        train_df = pd.DataFrame(train_data)
        train_output_path = os.path.join(OUTPUT_DIR, "add23_train.parquet")
        train_df.to_parquet(train_output_path)
        print(f"\nSaved {len(train_df)} train rows to {train_output_path}")
        print(f"Train label distribution:")
        print(train_df['label'].value_counts())
    
    # Save dev to parquet file
    if dev_data:
        dev_df = pd.DataFrame(dev_data)
        dev_output_path = os.path.join(OUTPUT_DIR, "add23_val.parquet")
        dev_df.to_parquet(dev_output_path)
        print(f"\nSaved {len(dev_df)} dev rows to {dev_output_path}")
        print(f"Dev label distribution:")
        print(dev_df['label'].value_counts())
    
    # Save eval R1 to parquet file
    if eval_r1_data:
        eval_r1_df = pd.DataFrame(eval_r1_data)
        eval_r1_output_path = os.path.join(OUTPUT_DIR, "add23_test_R1.parquet")
        eval_r1_df.to_parquet(eval_r1_output_path)
        print(f"\nSaved {len(eval_r1_df)} eval R1 rows to {eval_r1_output_path}")
        print(f"Eval R1 label distribution:")
        print(eval_r1_df['label'].value_counts())
    
    # Save eval R2 to parquet file
    if eval_r2_data:
        eval_r2_df = pd.DataFrame(eval_r2_data)
        eval_r2_output_path = os.path.join(OUTPUT_DIR, "add23_test_R2.parquet")
        eval_r2_df.to_parquet(eval_r2_output_path)
        print(f"\nSaved {len(eval_r2_df)} eval R2 rows to {eval_r2_output_path}")
        print(f"Eval R2 label distribution:")
        print(eval_r2_df['label'].value_counts())

if __name__ == "__main__":
    main()
