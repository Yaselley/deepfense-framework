import pandas as pd
import os
from glob import glob

# Configuration
REPLAYDF_ROOT = "/ds-slt/audio/yelkheir/ReplayDF/wav"
OUTPUT_DIR = "/netscratch/yelkheir/DeepFense/DeepFense/deepfense/config/parquets/ReplayDF"

# Languages found in the dataset
LANGUAGES = ["de", "en", "es", "fr", "it", "pl"]

def process_replaydf_language(language, dataset_name="ReplayDF"):
    """Process ReplayDF data for a specific language"""
    print(f"Processing ReplayDF {language} data...")
    
    data = []
    
    # Find all bonafide files for this language
    # Pattern: {speaker_id}/benign/{language}/*.wav
    benign_pattern = os.path.join(REPLAYDF_ROOT, "*", "benign", language, "*.wav")
    benign_files = glob(benign_pattern)
    
    for file_path in benign_files:
        # Extract relative path for ID (e.g., speaker_id/benign/language/filename.wav)
        rel_path = os.path.relpath(file_path, REPLAYDF_ROOT)
        # Use filename without extension as ID, or full relative path
        audio_id = os.path.splitext(os.path.basename(file_path))[0]
        
        data.append({
            "ID": audio_id,
            "path": file_path,
            "label": "bonafide",
            "dataset_name": f"{dataset_name}_{language}",
            "language": language
        })
    
    print(f"  Found {len(benign_files)} bonafide files")
    
    # Find all spoof files for this language
    # Pattern: {speaker_id}/spoof/{method}/{language}/*.wav
    spoof_pattern = os.path.join(REPLAYDF_ROOT, "*", "spoof", "*", language, "*.wav")
    spoof_files = glob(spoof_pattern)
    
    for file_path in spoof_files:
        # Extract relative path for ID
        rel_path = os.path.relpath(file_path, REPLAYDF_ROOT)
        # Use filename without extension as ID
        audio_id = os.path.splitext(os.path.basename(file_path))[0]
        
        data.append({
            "ID": audio_id,
            "path": file_path,
            "label": "spoof",
            "dataset_name": f"{dataset_name}_{language}",
            "language": language
        })
    
    print(f"  Found {len(spoof_files)} spoof files")
    print(f"  Total: {len(data)} entries for {language}")
    
    return data

def process_replaydf_all():
    """Process ReplayDF data for all languages combined"""
    print(f"Processing ReplayDF all languages combined...")
    
    all_data = []
    
    for language in LANGUAGES:
        lang_data = process_replaydf_language(language, dataset_name="ReplayDF")
        all_data.extend(lang_data)
    
    print(f"Total combined entries: {len(all_data)}")
    return all_data

def main():
    # Process each language separately
    for language in LANGUAGES:
        lang_data = process_replaydf_language(language)
        
        if lang_data:
            lang_df = pd.DataFrame(lang_data)
            lang_output_path = os.path.join(OUTPUT_DIR, f"replaydf_{language}_eval.parquet")
            lang_df.to_parquet(lang_output_path)
            print(f"\nSaved {len(lang_df)} {language} rows to {lang_output_path}")
            print(f"{language} label distribution:")
            print(lang_df['label'].value_counts())
    
    # Process all languages combined
    all_data = process_replaydf_all()
    
    if all_data:
        all_df = pd.DataFrame(all_data)
        all_output_path = os.path.join(OUTPUT_DIR, "replaydf_all_eval.parquet")
        all_df.to_parquet(all_output_path)
        print(f"\nSaved {len(all_df)} total rows to {all_output_path}")
        print(f"Overall label distribution:")
        print(all_df['label'].value_counts())
        print(f"\nLanguage distribution:")
        print(all_df['language'].value_counts())

if __name__ == "__main__":
    main()
