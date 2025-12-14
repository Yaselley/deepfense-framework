import pandas as pd
import os

# Configuration
ITW_EVAL_TXT = "/netscratch/yelkheir/datasets/ITW/eval.final.txt"
OUTPUT_DIR = "/netscratch/yelkheir/DeepFense/DeepFense/deepfense/config/parquets/InTheWild"

# Root directory for audio files
ITW_EVAL_ROOT = "/ds/audio/InTheWild/release_in_the_wild"

def process_itw_eval():
    """Process InTheWild eval data from eval.final.txt"""
    print(f"Processing InTheWild eval data...")
    
    data = []
    with open(ITW_EVAL_TXT, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            
            # Format: INDEX INDEX _ _ LABEL
            # Example: 0 0 _ _ spoof
            audio_id = parts[1]  # Second column is the audio ID
            label = parts[-1]   # Label is last column
            
            # Sanity check on label
            if label not in ["bonafide", "spoof"]:
                continue
            
            # Audio files are named as {audio_id}.wav
            file_path = os.path.join(ITW_EVAL_ROOT, f"{audio_id}.wav")
            
            data.append({
                "ID": audio_id,
                "path": file_path,
                "label": label,
                "dataset_name": "InTheWild"
            })
    
    print(f"Processed {len(data)} InTheWild entries")
    return data

def main():
    # Process InTheWild eval
    itw_data = process_itw_eval()
    
    # Save to parquet file
    if itw_data:
        itw_df = pd.DataFrame(itw_data)
        itw_output_path = os.path.join(OUTPUT_DIR, "itw_eval.parquet")
        itw_df.to_parquet(itw_output_path)
        print(f"\nSaved {len(itw_df)} InTheWild rows to {itw_output_path}")
        print(f"InTheWild label distribution:")
        print(itw_df['label'].value_counts())
    else:
        print("No data to save!")

if __name__ == "__main__":
    main()
