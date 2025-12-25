import pandas as pd
import argparse
from pathlib import Path

def process_wavefake_dataset(data_root):
    """
    Process WaveFake dataset by scanning all audio files.
    The dataset contains:
    - All files in subdirectories are spoof audios
    - Sub dir names are attacker types
    https://github.com/RUB-SysSec/WaveFake
    
    Data structure tree for data_root (/mount/arbeitsdaten54/projekte/deepfake/fad/data/wavefake/processed):
    data_root/
    └── generated_audio/
        ├── common_voices_prompts_from_conformer_fastspeech2_pwg_ljspeech/
        │   └── *.wav (audio files)
        ├── jsut_multi_band_melgan/
        │   └── *.wav (audio files)
        ├── jsut_parallel_wavegan/
        │   └── *.wav (audio files)
        ├── ljspeech_full_band_melgan/
        │   └── *.wav (audio files)
        ├── ljspeech_hifiGAN/
        │   └── *.wav (audio files)
        ├── ljspeech_melgan/
        │   └── *.wav (audio files)
        ├── ljspeech_melgan_large/
        │   └── *.wav (audio files)
        ├── ljspeech_multi_band_melgan/
        │   └── *.wav (audio files)
        ├── ljspeech_parallel_wavegan/
        │   └── *.wav (audio files)
        └── ljspeech_waveglow/
            └── *.wav (audio files)
    
    Args:
        data_root: root directory containing the uncompressed WaveFake dataset (generated_audio/ subdirectory will be appended)
    
    Returns:
        List of dicts containing ID, path, label, and dataset_name
    """
    data_root = Path(data_root)
    if not data_root.exists():
        print(f"Error: Data root directory not found: {data_root}")
        return []
    
    all_data = []
    
    for attacker_dir in data_root.iterdir():
        if not attacker_dir.is_dir():
            continue
        
        for wav_file in attacker_dir.rglob("*.wav"):
            audio_id = wav_file.stem
            
            all_data.append({
                "ID": audio_id,
                "path": str(wav_file.relative_to(data_root)),
                "label": "spoof",
                "dataset_name": "WaveFake",
            })
    
    return all_data

def main():
    parser = argparse.ArgumentParser(
        description="Generate parquet files for WaveFake dataset. It requires the uncompressed dataset root as input, the default output dir is the same directory as this script.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
    )
    
    output_dir = Path(__file__).parent.absolute()
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(output_dir),
    )
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    print(f"Writing parquet files to {output_dir}") 
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data_root = Path(args.data_root)
    data_root = data_root / "generated_audio"
    all_data = process_wavefake_dataset(data_root)
    
    if all_data:
        df = pd.DataFrame(all_data)
        output_path = output_dir / "wavefake.parquet"
        df.to_parquet(output_path)
        print(f"\nSaved {len(df)} rows to {output_path}")
        print(f"Label distribution:")
        print(df['label'].value_counts())
    else:
        print("No data to save!")

if __name__ == "__main__":
    main()
