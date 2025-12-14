import pandas as pd
import os

# Configuration
FOR_EVAL_TXT = "/netscratch/yelkheir/datasets/FoR/eval.final.txt"
OUTPUT_DIR = "/netscratch/yelkheir/DeepFense/DeepFense/deepfense/config/parquets/FoR"

# Root directory for audio files
FOR_EVAL_ROOT = "/ds-slt/audio/yelkheir/FoR/for-norm/testing"

def convert_filename(eval_filename):
    """
    Convert filename from eval format to actual file format.
    Eval format: file766_16k_norm_mono_silence
    Actual format: file766.wav_16k.wav_norm.wav_mono.wav_silence.wav
    """
    # Replace underscores with .wav_ except the first one after 'file'
    parts = eval_filename.split('_')
    if len(parts) > 1:
        # First part is like 'file766', rest are '16k', 'norm', 'mono', 'silence'
        base = parts[0]  # file766
        suffix_parts = parts[1:]  # ['16k', 'norm', 'mono', 'silence']
        # Convert to: file766.wav_16k.wav_norm.wav_mono.wav_silence.wav
        converted = base + '.wav_' + '.wav_'.join(suffix_parts) + '.wav'
        return converted
    return eval_filename

def process_for_eval():
    """Process FoR eval data from eval.final.txt"""
    print(f"Processing FoR eval data...")
    
    data = []
    with open(FOR_EVAL_TXT, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            
            # Format: FILENAME FILENAME _ _ LABEL
            # Example: file766_16k_norm_mono_silence file766_16k_norm_mono_silence _ _ spoof
            eval_filename = parts[1]  # Second column is the filename
            label = parts[-1]  # Label is last column
            
            # Sanity check on label
            if label not in ["bonafide", "spoof"]:
                continue
            
            # Convert filename format
            actual_filename = convert_filename(eval_filename)
            
            # Determine subdirectory based on label
            subdir = "fake" if label == "spoof" else "real"
            
            # Construct full path
            file_path = os.path.join(FOR_EVAL_ROOT, subdir, actual_filename)
            
            # Use the eval filename as ID (original format)
            data.append({
                "ID": eval_filename,
                "path": file_path,
                "label": label,
                "dataset_name": "FoR"
            })
    
    print(f"Processed {len(data)} FoR entries")
    return data

def main():
    # Process FoR eval
    for_data = process_for_eval()
    
    # Save to parquet file
    if for_data:
        for_df = pd.DataFrame(for_data)
        for_output_path = os.path.join(OUTPUT_DIR, "for_eval.parquet")
        for_df.to_parquet(for_output_path)
        print(f"\nSaved {len(for_df)} FoR rows to {for_output_path}")
        print(f"FoR label distribution:")
        print(for_df['label'].value_counts())
    else:
        print("No data to save!")

if __name__ == "__main__":
    main()
