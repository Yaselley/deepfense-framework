import pandas as pd
import os

# Configuration
ENVSSD_TRAIN_TXT = "/netscratch/yelkheir/DeepFense/Distillation/protocols/envssd/train.txt"
ENVSSD_VALID_TXT = "/netscratch/yelkheir/DeepFense/Distillation/protocols/envssd/valid.txt"
ENVSSD_EVAL_TXT = "/netscratch/yelkheir/DeepFense/Distillation/protocols/envssd/eval.txt"
OUTPUT_DIR = "/netscratch/yelkheir/DeepFense/DeepFense/deepfense/config/parquets/EnvSSD"

# Root directories for audio files
ENVSSD_TRAIN_ROOT = "/ds-slt/audio/yelkheir/EnvSSD/development"
ENVSSD_VALID_ROOT = "/ds-slt/audio/yelkheir/EnvSSD/development"
ENVSSD_EVAL_ROOT = "/ds-slt/audio/yelkheir/CodecFake/A3_fake"

def process_envssd_protocol(txt_file, root_dir, split_name, dataset_name="EnvSSD"):
    """Process EnvSSD protocol file (format: relative_path relative_path ... label)"""
    print(f"Processing {split_name} data from {txt_file}...")
    
    data = []
    with open(txt_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            
            # Format: relative_path relative_path ... label
            # Example: real_audio/UrbanSound8K/157867-8-0-2 real_audio/UrbanSound8K/157867-8-0-2 real real real
            # Or: 0 0 _ _ spoof (for eval.txt)
            
            # Get relative path from first column
            relative_path = parts[0]
            label = parts[-1]  # Label is last column
            
            # Map labels: "real" -> "bonafide", "fake" -> "spoof"
            if label == "real":
                label = "bonafide"
            elif label == "fake":
                label = "spoof"
            elif label not in ["bonafide", "spoof"]:
                continue
            
            # Construct full file path
            # For eval.txt, the format is different (numeric IDs like "0 0 _ _ spoof")
            if relative_path.isdigit():
                # For eval format, use the ID as filename with .wav extension
                audio_id = relative_path
                file_path = os.path.join(root_dir, f"{audio_id}.wav")
            else:
                # For train/valid format, use relative path with .wav extension if needed
                if not relative_path.endswith(('.wav', '.flac')):
                    relative_path = relative_path + ".wav"
                file_path = os.path.join(root_dir, relative_path)
                audio_id = relative_path
            
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
    train_data = process_envssd_protocol(
        ENVSSD_TRAIN_TXT,
        ENVSSD_TRAIN_ROOT,
        "train"
    )
    
    # Process valid split
    valid_data = process_envssd_protocol(
        ENVSSD_VALID_TXT,
        ENVSSD_VALID_ROOT,
        "valid"
    )
    
    # Process eval split
    eval_data = process_envssd_protocol(
        ENVSSD_EVAL_TXT,
        ENVSSD_EVAL_ROOT,
        "eval"
    )
    
    # Save train to parquet file
    if train_data:
        train_df = pd.DataFrame(train_data)
        train_output_path = os.path.join(OUTPUT_DIR, "envssd_train.parquet")
        train_df.to_parquet(train_output_path)
        print(f"\nSaved {len(train_df)} train rows to {train_output_path}")
        print(f"Train label distribution:")
        print(train_df['label'].value_counts())
    
    # Save valid to parquet file
    if valid_data:
        valid_df = pd.DataFrame(valid_data)
        valid_output_path = os.path.join(OUTPUT_DIR, "envssd_val.parquet")
        valid_df.to_parquet(valid_output_path)
        print(f"\nSaved {len(valid_df)} valid rows to {valid_output_path}")
        print(f"Valid label distribution:")
        print(valid_df['label'].value_counts())
    
    # Save eval to parquet file
    if eval_data:
        eval_df = pd.DataFrame(eval_data)
        eval_output_path = os.path.join(OUTPUT_DIR, "envssd_eval.parquet")
        eval_df.to_parquet(eval_output_path)
        print(f"\nSaved {len(eval_df)} eval rows to {eval_output_path}")
        print(f"Eval label distribution:")
        print(eval_df['label'].value_counts())

if __name__ == "__main__":
    main()

