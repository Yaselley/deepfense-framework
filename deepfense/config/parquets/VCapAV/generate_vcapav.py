import pandas as pd
import argparse
from pathlib import Path

def process_vcapav_dataset(data_root):
    """
    Process VCapAV dataset by scanning all audio files.
    Data structure tree for data_root 
    data_root/
    ├── T2A/ (spoof - Text-to-Audio)
    │   ├── audiocraft_audio_cut/
    │   │   └── *.wav (audio files)
    │   ├── audioLDM1_audio_cut/
    │   │   └── *.wav (audio files)
    │   └── audioLDM2_audio_cut/
    │       └── *.wav (audio files)
    ├── V2A/ (spoof - Video-to-Audio)
    │   ├── SMIIPdata2/
    │   │   └── datasets/
    │   │       └── V2A/
    │   │           ├── V2A-Mapper/
    │   │           │   └── V2A_audio_cut/
    │   │           │       └── *.wav (audio files)
    │   │           └── V2A-MLP/
    │   │               └── YMJ_audio_cut/
    │   │                   └── *.wav (audio files)
    │   └── home/ (skipped, contains videos)
    └── VGGsound_test_14923_audio_cut/ (bonafide)
        └── VGGsound_test_15446_audio_cut/
            └── *.wav (audio files)
    
    Note: The paper mentions three partitions (dev1, dev2, dev3), but the metadata
    is not provided in the dataset, so we create a single parquet file with all data.
    
    Note: The same filename can appear in multiple directories (e.g., different generation methods),
    so we use relative path from data_root as the ID.
    
    Reference: https://github.com/wailywang/VCapAV/
    
    Args:
        data_root: root directory containing the processed VCapAV dataset
    
    Returns:
        List of dicts containing ID, path, label, and dataset_name
    """
    data_root = Path(data_root)
    if not data_root.exists():
        print(f"Error: Data root directory not found: {data_root}")
        return []
    
    all_data = []
    
    t2a_dir = data_root / "T2A"
    audio_id = 0

    if t2a_dir.exists():
        for wav_file in t2a_dir.rglob("*.wav"):
            audio_id += 1
            relative_path = wav_file.relative_to(data_root)
            
            all_data.append({
                "ID": str(audio_id),
                "path": str(relative_path),
                "label": "spoof",
                "dataset_name": "VCapAV"
            })
    
    v2a_dir = data_root / "V2A"
    if v2a_dir.exists():
        for wav_file in v2a_dir.rglob("*.wav"):
            audio_id += 1
            relative_path = wav_file.relative_to(data_root)
            
            all_data.append({
                "ID": str(audio_id),
                "path": str(relative_path),
                "label": "spoof",
                "dataset_name": "VCapAV"
            })
    
    vggsound_dir = data_root / "VGGsound_test_14923_audio_cut"
    if vggsound_dir.exists():
        for wav_file in vggsound_dir.rglob("*.wav"):
            relative_path = wav_file.relative_to(data_root)
            audio_id += 1
            
            all_data.append({
                "ID": str(audio_id),
                "path": str(relative_path),
                "label": "bonafide",
                "dataset_name": "VCapAV"
            })
    
    return all_data

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
    )
    
    parser.add_argument(
        "--meta_root",
        type=str,
        default=None,
    )
    
    output_dir = Path(__file__).parent.absolute()
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(output_dir),
    )
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    print(f"Writing parquet file to {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"Error: Data root directory not found: {data_root}")
        return
    
    print(f"\nProcessing VCapAV dataset from {data_root}...")
    all_data = process_vcapav_dataset(data_root)
    
    if all_data:
        df = pd.DataFrame(all_data)
        output_path = output_dir / "vcapav.parquet"
        df.to_parquet(output_path)
        print(f"\nSaved {len(df)} rows to {output_path}")
        print(f"Label distribution:")
        print(df['label'].value_counts())
    else:
        print("No data to save!")

if __name__ == "__main__":
    main()

