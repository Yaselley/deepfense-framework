import pandas as pd
import os

# Configuration
PROTOCOL_DIR = "/netscratch/yelkheir/DeepFense/Distillation/protocols/asv19"
OUTPUT_DIR = "/netscratch/yelkheir/DeepFense/DeepFense/deepfense/config/parquets"

# Placeholder roots - User should update these or we assume a structure
# Since I was told not to look for files, I will construct paths assuming a standard structure.
# If the user has a specific mount point, they can replace these roots.
TRAIN_ROOT = "/ds/audio/LA_19/ASVspoof2019_LA_train/flac"
DEV_ROOT = "/ds/audio/LA_19/ASVspoof2019_LA_dev/flac"
EVAL_ROOT = "/ds/audio/LA_19/ASVspoof2019_LA_eval/flac"

def process_protocol(txt_file, root_dir, output_name, dataset_name="ASVSpoof19"):
    print(f"Processing {txt_file}...")
    
    data = []
    with open(txt_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            
            # Format: SPEAKER_ID AUDIO_ID ENVIRONMENT SYSTEM_ID LABEL
            # Example: LA_0079 LA_T_1138215 - - bonafide
            # Eval might be slightly different sometimes, let's check parts length
            
            audio_id = parts[1]
            label = parts[-1] # Label is usually last
            
            # Sanity check on label
            if label not in ["bonafide", "spoof"]:
                # Some protocols might have different structure, but ASV19 LA usually follows this.
                # Eval file example from earlier: LA_0039 LA_E_2834763 - A11 spoof
                pass

            file_path = os.path.join(root_dir, f"{audio_id}.flac")
            
            data.append({
                "ID": audio_id,
                "path": file_path,
                "label": label,
                "dataset_name": dataset_name
            })
            
    df = pd.DataFrame(data)
    output_path = os.path.join(OUTPUT_DIR, output_name)
    df.to_parquet(output_path)
    print(f"Saved {len(df)} rows to {output_path}")

def main():
    # 1. Train
    process_protocol(
        os.path.join(PROTOCOL_DIR, "asvspoof19.txt"),
        TRAIN_ROOT,
        "asvspoof19_train.parquet"
    )
    
    # 2. Dev (Validation)
    process_protocol(
        os.path.join(PROTOCOL_DIR, "asvspoof19.dev.txt"),
        DEV_ROOT,
        "asvspoof19_val.parquet"
    )
    
    # 3. Eval (Test)
    process_protocol(
        os.path.join(PROTOCOL_DIR, "asvspoof19.eval.txt"),
        EVAL_ROOT,
        "asvspoof19_test.parquet"
    )

if __name__ == "__main__":
    main()

