import pandas as pd
import os
from pathlib import Path

# Configuration
FAKEMUSIC_TRAIN_TXT = "/netscratch/yelkheir/DeepFense/Distillation/protocols/fakemusic/train.txt"
FAKEMUSIC_DEV_TXT = "/netscratch/yelkheir/DeepFense/Distillation/protocols/fakemusic/dev.txt"
FAKEMUSIC_EVAL_TXT = "/netscratch/yelkheir/DeepFense/Distillation/protocols/fakemusic/eval.txt"
OUTPUT_DIR = "/netscratch/yelkheir/DeepFense/DeepFense/deepfense/config/parquets/FakeMusicCaps"

# Root directory for audio files (from config.py)
FAKEMUSIC_ROOT = "/ds-slt/audio/yelkheir/FakeMusicCaps"

def process_fakemusic_protocol(txt_file, root_dir, split_name, dataset_name="FakeMusicCaps"):
    """Process FakeMusicCaps protocol file (format: ID ID _ _ LABEL)"""
    print(f"Processing {split_name} data from {txt_file}...")
    
    data = []
    skipped_count = 0
    with open(txt_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            
            # Format: ID ID _ _ LABEL
            # Example for bonafide: dS8AZdmn8Wk dS8AZdmn8Wk _ _ bonafide
            # Example for spoof: MusicGen_medium/r4JRDHYukZ4 MusicGen_medium/r4JRDHYukZ4 _ _ spoof
            # The ID can include subdirectory path (e.g., MusicGen_medium/r4JRDHYukZ4)
            audio_id = parts[0]  # First column is the audio ID (includes path if applicable)
            label = parts[-1]    # Label is last column
            
            # Sanity check on label
            if label not in ["bonafide", "spoof"]:
                continue
            
            # Audio files are named as {audio_id}.wav
            # Example: /ds-slt/audio/yelkheir/FakeMusicCaps/dS8AZdmn8Wk.wav (bonafide)
            # Example: /ds-slt/audio/yelkheir/FakeMusicCaps/MusicGen_medium/r4JRDHYukZ4.wav (spoof)
            file_path = os.path.join(root_dir, f"{audio_id}.wav")
            
            # Only add entry if file exists
            if os.path.exists(file_path):
                data.append({
                    "ID": audio_id,
                    "path": file_path,
                    "label": label,
                    "dataset_name": dataset_name
                })
            else:
                skipped_count += 1
    
    print(f"Processed {len(data)} {split_name} entries (skipped {skipped_count} missing files)")
    return data

def main():
    # Process train split
    train_data = process_fakemusic_protocol(
        FAKEMUSIC_TRAIN_TXT,
        FAKEMUSIC_ROOT,
        "train"
    )
    
    # Process dev split
    dev_data = process_fakemusic_protocol(
        FAKEMUSIC_DEV_TXT,
        FAKEMUSIC_ROOT,
        "dev"
    )
    
    # Process eval split
    eval_data = process_fakemusic_protocol(
        FAKEMUSIC_EVAL_TXT,
        FAKEMUSIC_ROOT,
        "eval"
    )
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save train to parquet file
    if train_data:
        train_df = pd.DataFrame(train_data)
        train_output_path = os.path.join(OUTPUT_DIR, "fakemusiccaps_train.parquet")
        train_df.to_parquet(train_output_path)
        print(f"\nSaved {len(train_df)} train rows to {train_output_path}")
        print(f"Train label distribution:")
        print(train_df['label'].value_counts())
    
    # Save dev to parquet file
    if dev_data:
        dev_df = pd.DataFrame(dev_data)
        dev_output_path = os.path.join(OUTPUT_DIR, "fakemusiccaps_dev.parquet")
        dev_df.to_parquet(dev_output_path)
        print(f"\nSaved {len(dev_df)} dev rows to {dev_output_path}")
        print(f"Dev label distribution:")
        print(dev_df['label'].value_counts())
    
    # Save eval to parquet file
    if eval_data:
        eval_df = pd.DataFrame(eval_data)
        eval_output_path = os.path.join(OUTPUT_DIR, "fakemusiccaps_eval.parquet")
        eval_df.to_parquet(eval_output_path)
        print(f"\nSaved {len(eval_df)} eval rows to {eval_output_path}")
        print(f"Eval label distribution:")
        print(eval_df['label'].value_counts())

if __name__ == "__main__":
    main()
