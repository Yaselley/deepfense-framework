import pandas as pd
import os

# Configuration
DFADD_EVAL_TXT = "/netscratch/yelkheir/datasets/DFADD/eval.txt"
OUTPUT_DIR = "/netscratch/yelkheir/DeepFense/DeepFense/deepfense/config/parquets/DFADD"

# Root directory for audio files
DFADD_EVAL_ROOT = "/ds-slt/audio/yelkheir/DFADD/audios"

def process_dfadd_eval():
    """Process DFADD eval data from eval.txt"""
    print(f"Processing DFADD eval data...")
    
    data = []
    with open(DFADD_EVAL_TXT, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            
            # Format: ID ID - - LABEL
            # Example: p227_164_GradTTS p227_164_GradTTS - - spoof
            audio_id = parts[1]  # Second column is the audio ID
            label = parts[-1]   # Label is last column
            
            # Sanity check on label
            if label not in ["bonafide", "spoof"]:
                continue
            
            # Audio files are named as {audio_id}.wav
            file_path = os.path.join(DFADD_EVAL_ROOT, f"{audio_id}.wav")
            
            data.append({
                "ID": audio_id,
                "path": file_path,
                "label": label,
                "dataset_name": "DFADD"
            })
    
    print(f"Processed {len(data)} DFADD entries")
    return data

def main():
    # Process DFADD eval
    eval_data = process_dfadd_eval()
    
    # Save to parquet file
    if eval_data:
        eval_df = pd.DataFrame(eval_data)
        eval_output_path = os.path.join(OUTPUT_DIR, "dfadd_eval.parquet")
        eval_df.to_parquet(eval_output_path)
        print(f"\nSaved {len(eval_df)} DFADD rows to {eval_output_path}")
        print(f"DFADD label distribution:")
        print(eval_df['label'].value_counts())
    else:
        print("No data to save!")

if __name__ == "__main__":
    main()
