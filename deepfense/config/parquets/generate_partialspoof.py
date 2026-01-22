import pandas as pd
import os

# Define source and target paths
source_dir = "/netscratch/yelkheir/datasets/partial_asvspoof"
output_dir = "DeepFense/DeepFense/deepfense/config/parquets/PartialSpoof"
os.makedirs(output_dir, exist_ok=True)

# Define dataset splits and their corresponding filenames
splits = {
    "train": "train.txt",
    "dev": "dev.txt",
    "eval": "eval.txt"
}

def process_split(split_name, filename):
    input_path = os.path.join(source_dir, filename)
    if not os.path.exists(input_path):
        print(f"Warning: {input_path} not found. Skipping.")
        return

    print(f"Processing {split_name} from {input_path}...")
    
    data = []
    with open(input_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
                
            # Structure: SPEAKER_ID AUDIO_ID - SYSTEM_ID KEY
            # Index:     0          1        2 3         4
            audio_id = parts[1]
            key = parts[4] # 'spoof' or 'bonafide'
            
            # Assuming flac extension for now, can be adjusted if needed
            # Storing relative path (filename.flac)
            # The root_dir in config will handle the rest
            file_path = f"{audio_id}.flac" 
            
            if split_name == "train":
                file_path = f"/ds/audio/partialSpoof/database/train/con_wav/{audio_id}.wav"
            elif split_name == "dev":
                file_path = f"/ds/audio/partialSpoof/database/dev/con_wav/{audio_id}.wav"
            elif split_name == "eval":
                file_path = f"/ds/audio/partialSpoof/database/eval/con_wav/{audio_id}.wav"

            data.append({
                "path": file_path,
                "label": key,
                "ID": audio_id,
                "speaker": parts[0],
                "system_id": parts[3]
            })

    df = pd.DataFrame(data)
    
    # Save to parquet
    output_path = os.path.join(output_dir, f"{split_name}.parquet")
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")

for split, filename in splits.items():
    process_split(split, filename)
