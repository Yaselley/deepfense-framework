import pandas as pd
import os

# Configuration
ADD22_TRAIN_TXT = "/netscratch/yelkheir/datasets/add_2022/train.txt"
ADD22_DEV_TXT = "/netscratch/yelkheir/datasets/add_2022/dev.txt"
ADD22_EVAL_TRACK1_TXT = "/netscratch/yelkheir/datasets/add_2022/eval_track1.txt"
ADD22_EVAL_TRACK3_TXT = "/netscratch/yelkheir/datasets/add_2022/eval_track3.txt"
OUTPUT_DIR = "/netscratch/yelkheir/DeepFense/DeepFense/deepfense/config/parquets/ADD22"

# Root directories for audio files (need to be verified/updated)
# Based on naming convention: ADD_T_* for train, ADD_D_* for dev, ADD_E1_* for track1, ADD_E3_* for track3
ADD22_TRAIN_ROOT = "/ds/audio/add"  # Update if different
ADD22_DEV_ROOT = "/ds/audio/add"    # Update if different
ADD22_EVAL_TRACK1_ROOT = "/ds/audio/add"  # Update if different
ADD22_EVAL_TRACK3_ROOT = "/ds/audio/add"  # Update if different

def process_add22_train_dev(txt_file, root_dir, split_name, dataset_name="ADD22"):
    """Process ADD22 train or dev file (format: filename.wav label)"""
    print(f"Processing {split_name} data from {txt_file}...")
    
    data = []
    with open(txt_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            
            # Format: filename.wav label
            # Example: ADD_T_00000000.wav genuine
            filename = parts[0]
            label = parts[1]
            
            # Convert "genuine" to "bonafide" for consistency
            if label == "genuine":
                label = "bonafide"
            
            # Sanity check on label
            if label not in ["bonafide", "spoof"]:
                continue
            
            # Construct file path
            file_path = os.path.join(root_dir, filename)
            
            # Use filename without extension as ID
            audio_id = os.path.splitext(filename)[0]
            
            data.append({
                "ID": audio_id,
                "path": file_path,
                "label": label,
                "dataset_name": dataset_name
            })
    
    print(f"Processed {len(data)} {split_name} entries")
    return data

def process_add22_eval_track1(txt_file, root_dir, split_name, dataset_name="ADD22"):
    """Process ADD22 eval track1 file (format: ID label)"""
    print(f"Processing {split_name} data from {txt_file}...")
    
    data = []
    with open(txt_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            
            # Format: ID label
            # Example: ADD_E1_00000000 spoof
            audio_id = parts[0]
            label = parts[1]
            
            # Sanity check on label
            if label not in ["bonafide", "spoof"]:
                continue
            
            # Construct file path (assuming .wav extension)
            file_path = os.path.join(root_dir, f"{audio_id}.wav")
            
            data.append({
                "ID": audio_id,
                "path": file_path,
                "label": label,
                "dataset_name": dataset_name
            })
    
    print(f"Processed {len(data)} {split_name} entries")
    return data

def process_add22_eval_track3(txt_file, root_dir, split_name, dataset_name="ADD22"):
    """Process ADD22 eval track3 file (format: filename.wav label number)"""
    print(f"Processing {split_name} data from {txt_file}...")
    
    data = []
    with open(txt_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            
            # Format: filename.wav label number
            # Example: ADD_E3_00000000.wav spoof 0
            filename = parts[0]
            label = parts[1]
            
            # Sanity check on label
            if label not in ["bonafide", "spoof"]:
                continue
            
            # Construct file path
            file_path = os.path.join(root_dir, filename)
            
            # Use filename without extension as ID
            audio_id = os.path.splitext(filename)[0]
            
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
    train_data = process_add22_train_dev(
        ADD22_TRAIN_TXT,
        ADD22_TRAIN_ROOT,
        "train"
    )
    
    # Process dev split
    dev_data = process_add22_train_dev(
        ADD22_DEV_TXT,
        ADD22_DEV_ROOT,
        "dev"
    )
    
    # Process eval track1
    eval_track1_data = process_add22_eval_track1(
        ADD22_EVAL_TRACK1_TXT,
        ADD22_EVAL_TRACK1_ROOT,
        "eval.track1"
    )
    
    # Process eval track3
    eval_track3_data = process_add22_eval_track3(
        ADD22_EVAL_TRACK3_TXT,
        ADD22_EVAL_TRACK3_ROOT,
        "eval.track3"
    )
    
    # Save train to parquet file
    if train_data:
        train_df = pd.DataFrame(train_data)
        train_output_path = os.path.join(OUTPUT_DIR, "add22_train.parquet")
        train_df.to_parquet(train_output_path)
        print(f"\nSaved {len(train_df)} train rows to {train_output_path}")
        print(f"Train label distribution:")
        print(train_df['label'].value_counts())
    
    # Save dev to parquet file
    if dev_data:
        dev_df = pd.DataFrame(dev_data)
        dev_output_path = os.path.join(OUTPUT_DIR, "add22_val.parquet")
        dev_df.to_parquet(dev_output_path)
        print(f"\nSaved {len(dev_df)} dev rows to {dev_output_path}")
        print(f"Dev label distribution:")
        print(dev_df['label'].value_counts())
    
    # Save eval track1 to parquet file
    if eval_track1_data:
        eval_track1_df = pd.DataFrame(eval_track1_data)
        eval_track1_output_path = os.path.join(OUTPUT_DIR, "add22_test_track1.parquet")
        eval_track1_df.to_parquet(eval_track1_output_path)
        print(f"\nSaved {len(eval_track1_df)} eval track1 rows to {eval_track1_output_path}")
        print(f"Eval track1 label distribution:")
        print(eval_track1_df['label'].value_counts())
    
    # Save eval track3 to parquet file
    if eval_track3_data:
        eval_track3_df = pd.DataFrame(eval_track3_data)
        eval_track3_output_path = os.path.join(OUTPUT_DIR, "add22_test_track3.parquet")
        eval_track3_df.to_parquet(eval_track3_output_path)
        print(f"\nSaved {len(eval_track3_df)} eval track3 rows to {eval_track3_output_path}")
        print(f"Eval track3 label distribution:")
        print(eval_track3_df['label'].value_counts())

if __name__ == "__main__":
    main()
