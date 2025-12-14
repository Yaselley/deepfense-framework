import pandas as pd
import os

# Configuration
CODECFAKE_EVAL_TXT = "/netscratch/yelkheir/datasets/codecfake/eval.txt"
OUTPUT_DIR = "/netscratch/yelkheir/DeepFense/DeepFense/deepfense/config/parquets/CodecFake"

# Root directory for audio files
CODECFAKE_EVAL_ROOT = "/ds-slt/audio/yelkheir/CodecFake/test"

def process_codecfake_eval():
    """Process CodecFake eval data from eval.txt"""
    print(f"Processing CodecFake eval data...")
    
    data = []
    with open(CODECFAKE_EVAL_TXT, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            
            # Format: ID ID - - LABEL
            # Example: C1/F01_SSB05440084 C1/F01_SSB05440084 - - spoof
            # The ID includes subdirectory path (e.g., C1/F01_SSB05440084)
            audio_id = parts[1]  # Second column is the audio ID (includes path)
            label = parts[-1]    # Label is last column
            
            # Sanity check on label
            if label not in ["bonafide", "spoof"]:
                continue
            
            # Audio files are named as {audio_id}.wav (audio_id includes subdirectory)
            # Example: /ds-slt/audio/yelkheir/CodecFake/test/C1/F01_SSB05440084.wav
            file_path = os.path.join(CODECFAKE_EVAL_ROOT, f"{audio_id}.wav")
            
            data.append({
                "ID": audio_id,
                "path": file_path,
                "label": label,
                "dataset_name": "CodecFake"
            })
    
    print(f"Processed {len(data)} CodecFake entries")
    return data

def main():
    # Process CodecFake eval
    eval_data = process_codecfake_eval()
    
    # Save to parquet file
    if eval_data:
        eval_df = pd.DataFrame(eval_data)
        eval_output_path = os.path.join(OUTPUT_DIR, "codecfake_eval.parquet")
        eval_df.to_parquet(eval_output_path)
        print(f"\nSaved {len(eval_df)} CodecFake rows to {eval_output_path}")
        print(f"CodecFake label distribution:")
        print(eval_df['label'].value_counts())
    else:
        print("No data to save!")

if __name__ == "__main__":
    main()
