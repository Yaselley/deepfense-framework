import pandas as pd
import os

# Configuration
LA21_EVAL_TXT = "/netscratch/yelkheir/datasets/asvsoof/la21.eval.txt"
DF21_EVAL_TXT = "/netscratch/yelkheir/datasets/asvsoof/df21.eval.txt"
LA21_TRIAL_METADATA = "/ds/audio/LA_21/keys/LA/CM/trial_metadata.txt"
OUTPUT_DIR = "/netscratch/yelkheir/DeepFense/DeepFense/deepfense/config/parquets"

# Root directories for audio files
LA21_EVAL_ROOT = "/ds/audio/LA_21/ASVspoof2021_LA_eval/flac"
DF21_EVAL_ROOT = "/ds/audio/DF_21/ASVspoof2021_DF_eval/flac"

def process_la21_eval():
    """Process LA21 eval data from la21.eval.txt and trial_metadata.txt"""
    print(f"Processing LA21 eval data...")
    
    # Read LA21 eval IDs
    la21_ids = set()
    with open(LA21_EVAL_TXT, 'r') as f:
        for line in f:
            audio_id = line.strip()
            if audio_id:
                la21_ids.add(audio_id)
    
    print(f"Found {len(la21_ids)} LA21 eval IDs")
    
    # Read trial_metadata to get labels (only eval phase)
    # Format: SPEAKER_ID AUDIO_ID ... LABEL ... PHASE
    # Example: LA_0009 LA_E_9332881 alaw ita_tx A07 spoof notrim eval
    id_to_label = {}
    with open(LA21_TRIAL_METADATA, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8:
                audio_id = parts[1]  # Column 1 (0-indexed): Audio ID
                label = parts[5]     # Column 5 (0-indexed): Label (spoof/bonafide)
                phase = parts[7]     # Column 7 (0-indexed): Phase (eval/progress)
                
                # Only process eval phase entries
                if phase == "eval" and audio_id in la21_ids:
                    id_to_label[audio_id] = label
    
    print(f"Found labels for {len(id_to_label)} LA21 entries")
    
    # Create data entries (only for IDs with valid labels)
    data = []
    for audio_id in la21_ids:
        label = id_to_label.get(audio_id)
        if label and label in ["bonafide", "spoof"]:
            file_path = os.path.join(LA21_EVAL_ROOT, f"{audio_id}.flac")
            
            data.append({
                "ID": audio_id,
                "path": file_path,
                "label": label,
                "dataset_name": "ASVSpoof21_LA"
            })
    
    return data

def process_df21_eval():
    """Process DF21 eval data from df21.eval.txt"""
    print(f"Processing DF21 eval data...")
    
    data = []
    with open(DF21_EVAL_TXT, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            
            # Format: SPEAKER_ID AUDIO_ID - - LABEL
            # Example: LA_0023 DF_E_2000011 - - spoof
            audio_id = parts[1]
            label = parts[-1]  # Label is last column
            
            # Sanity check on label
            if label not in ["bonafide", "spoof"]:
                continue
            
            file_path = os.path.join(DF21_EVAL_ROOT, f"{audio_id}.flac")
            
            data.append({
                "ID": audio_id,
                "path": file_path,
                "label": label,
                "dataset_name": "ASVSpoof21_DF"
            })
    
    print(f"Processed {len(data)} DF21 entries")
    return data

def main():
    # Process LA21 eval
    la21_data = process_la21_eval()
    
    # Process DF21 eval
    df21_data = process_df21_eval()
    
    # Save LA21 to separate parquet file
    if la21_data:
        la21_df = pd.DataFrame(la21_data)
        la21_output_path = os.path.join(OUTPUT_DIR, "asvspoof21_la_eval.parquet")
        la21_df.to_parquet(la21_output_path)
        print(f"\nSaved {len(la21_df)} LA21 rows to {la21_output_path}")
        print(f"LA21 label distribution:")
        print(la21_df['label'].value_counts())
    
    # Save DF21 to separate parquet file
    if df21_data:
        df21_df = pd.DataFrame(df21_data)
        df21_output_path = os.path.join(OUTPUT_DIR, "asvspoof21_df_eval.parquet")
        df21_df.to_parquet(df21_output_path)
        print(f"\nSaved {len(df21_df)} DF21 rows to {df21_output_path}")
        print(f"DF21 label distribution:")
        print(df21_df['label'].value_counts())

if __name__ == "__main__":
    main()
