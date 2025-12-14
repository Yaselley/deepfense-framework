import pandas as pd
import os
from glob import glob
from collections import defaultdict

# Configuration
MLAAD_FAKE_ROOT = "/ds-slt/audio/MLAAD_v8/MLAAD/fake"
MLAAD_REAL_ROOT = "/ds-slt/audio/yelkheir/MLAAD/real"
OUTPUT_DIR = "/netscratch/yelkheir/DeepFense/DeepFense/deepfense/config/parquets/MLAAD"

# Language mapping: fake language code -> real language codes
# Some languages have multiple variants (e.g., en -> en_UK, en_US)
LANGUAGE_MAPPING = {
    "de": ["de_DE"],
    "en": ["en_UK", "en_US"],
    "es": ["es_ES"],
    "fr": ["fr_FR"],
    "it": ["it_IT"],
    "pl": ["pl_PL"],
    "ru": ["ru_RU"],
    "uk": ["uk_UK"],
}

def collect_all_real_files():
    """Collect all real files and create a mapping by filename"""
    print("Collecting all real files...")
    
    # Create a mapping: filename (without extension) -> file path
    # This allows us to quickly find matching real files for fake files
    real_files_map = {}
    
    # Find all wav files in real directory
    real_pattern = os.path.join(MLAAD_REAL_ROOT, "*", "**", "*.wav")
    all_real_files = glob(real_pattern, recursive=True)
    
    print(f"Found {len(all_real_files)} real files")
    
    for file_path in all_real_files:
        # Get filename without extension as key
        filename = os.path.basename(file_path)
        audio_id = os.path.splitext(filename)[0]
        
        # Store the file path (if multiple files with same name exist, keep the first one)
        if audio_id not in real_files_map:
            real_files_map[audio_id] = file_path
    
    print(f"Created mapping for {len(real_files_map)} unique filenames")
    
    return real_files_map

def collect_fake_files_by_language():
    """Collect all fake files organized by language"""
    print("\nCollecting all fake files...")
    
    fake_files_by_lang = defaultdict(list)
    
    # Get all language directories
    if not os.path.exists(MLAAD_FAKE_ROOT):
        print(f"Warning: {MLAAD_FAKE_ROOT} does not exist!")
        return fake_files_by_lang
    
    language_dirs = [d for d in os.listdir(MLAAD_FAKE_ROOT) 
                    if os.path.isdir(os.path.join(MLAAD_FAKE_ROOT, d))]
    
    for lang in language_dirs:
        lang_path = os.path.join(MLAAD_FAKE_ROOT, lang)
        
        # Find all wav files in this language directory (recursively)
        fake_pattern = os.path.join(lang_path, "**", "*.wav")
        lang_files = glob(fake_pattern, recursive=True)
        
        for file_path in lang_files:
            # Get filename without extension as ID
            audio_id = os.path.splitext(os.path.basename(file_path))[0]
            
            fake_files_by_lang[lang].append({
                "ID": audio_id,
                "path": file_path,
                "label": "spoof",
                "language": lang
            })
        
        if lang_files:
            print(f"  {lang}: {len(lang_files)} files")
    
    return fake_files_by_lang

def process_mlaad_language(lang, real_files_map, fake_files_by_lang):
    """Process MLAAD data for a specific language"""
    print(f"\nProcessing MLAAD {lang} data...")
    
    data = []
    matched_real_count = 0
    
    # Add fake files for this language
    if lang in fake_files_by_lang:
        fake_files = fake_files_by_lang[lang]
        data.extend(fake_files)
        print(f"  Fake files: {len(fake_files)}")
        
        # For each fake file, try to find matching real file
        for fake_entry in fake_files:
            fake_id = fake_entry["ID"]
            
            # Look for matching real file with same filename
            if fake_id in real_files_map:
                real_file_path = real_files_map[fake_id]
                # Create ID with "_real" suffix to avoid conflicts
                real_id = f"{fake_id}_real"
                
                data.append({
                    "ID": real_id,
                    "path": real_file_path,
                    "label": "bonafide",
                    "language": lang
                })
                matched_real_count += 1
    else:
        print(f"  Fake files: 0 (no fake files found for {lang})")
    
    print(f"  Matched real files: {matched_real_count}")
    print(f"  Total: {len(data)} entries for {lang}")
    
    return data

def main():
    # Step 1: Collect all real files (create filename mapping)
    real_files_map = collect_all_real_files()
    
    # Step 2: Collect all fake files
    fake_files_by_lang = collect_fake_files_by_language()
    
    # Step 3: Get all languages from fake files (since we match real files to fake files)
    all_languages = sorted(fake_files_by_lang.keys())
    
    print(f"\nFound {len(all_languages)} languages with fake files: {', '.join(all_languages)}")
    
    # Step 4: Process each language separately
    language_data = {}
    for lang in all_languages:
        lang_data = process_mlaad_language(lang, real_files_map, fake_files_by_lang)
        
        if lang_data:
            lang_df = pd.DataFrame(lang_data)
            # Normalize language name for filename (replace special chars)
            lang_name = lang.replace("-", "_")
            lang_output_path = os.path.join(OUTPUT_DIR, f"mlaad_{lang_name}_eval.parquet")
            lang_df.to_parquet(lang_output_path)
            print(f"\nSaved {len(lang_df)} {lang} rows to {lang_output_path}")
            print(f"{lang} label distribution:")
            print(lang_df['label'].value_counts())
            language_data[lang] = lang_data
    
    # Step 5: Create combined parquet file
    all_data = []
    for lang_data in language_data.values():
        all_data.extend(lang_data)
    
    if all_data:
        all_df = pd.DataFrame(all_data)
        all_output_path = os.path.join(OUTPUT_DIR, "mlaad_all_eval.parquet")
        all_df.to_parquet(all_output_path)
        print(f"\n{'='*60}")
        print(f"Saved {len(all_df)} total rows to {all_output_path}")
        print(f"Overall label distribution:")
        print(all_df['label'].value_counts())
        print(f"\nLanguage distribution:")
        print(all_df['language'].value_counts())

if __name__ == "__main__":
    main()
