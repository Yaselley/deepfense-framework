import pandas as pd
import os
from pathlib import Path

# Configuration
OUTPUT_DIR = "/netscratch/yelkheir/DeepFense/DeepFense/deepfense/config/parquets/FakeMusicCaps"

PARQUET_FILES = [
    "fakemusiccaps_train.parquet",
    "fakemusiccaps_dev.parquet",
    "fakemusiccaps_eval.parquet"
]

def clean_parquet_file(parquet_path):
    """Clean parquet file by removing rows where file_path doesn't exist"""
    print(f"\nProcessing: {parquet_path}")
    
    # Read parquet file
    df = pd.read_parquet(parquet_path)
    original_count = len(df)
    print(f"  Original rows: {original_count}")
    
    # Check if 'path' column exists
    if 'path' not in df.columns:
        print(f"  Warning: 'path' column not found in {parquet_path}")
        return
    
    # Check file existence
    df['file_exists'] = df['path'].apply(lambda x: os.path.exists(x) if pd.notna(x) else False)
    
    # Count missing files
    missing_count = (~df['file_exists']).sum()
    print(f"  Missing files: {missing_count}")
    
    # Filter out rows where file doesn't exist
    df_cleaned = df[df['file_exists']].copy()
    
    # Remove the temporary 'file_exists' column
    df_cleaned = df_cleaned.drop(columns=['file_exists'])
    
    cleaned_count = len(df_cleaned)
    print(f"  Cleaned rows: {cleaned_count}")
    print(f"  Removed rows: {original_count - cleaned_count}")
    
    # Show label distribution if label column exists
    if 'label' in df_cleaned.columns:
        print(f"  Label distribution:")
        print(df_cleaned['label'].value_counts())
    
    # Save cleaned parquet file
    df_cleaned.to_parquet(parquet_path, index=False)
    print(f"  Saved cleaned parquet to: {parquet_path}")
    
    return original_count, cleaned_count

def main():
    print("Cleaning FakeMusicCaps parquet files...")
    print(f"Output directory: {OUTPUT_DIR}")
    
    total_original = 0
    total_cleaned = 0
    
    for parquet_file in PARQUET_FILES:
        parquet_path = os.path.join(OUTPUT_DIR, parquet_file)
        
        if not os.path.exists(parquet_path):
            print(f"\nWarning: {parquet_path} does not exist, skipping...")
            continue
        
        try:
            original, cleaned = clean_parquet_file(parquet_path)
            total_original += original
            total_cleaned += cleaned
        except Exception as e:
            print(f"\nError processing {parquet_path}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total original rows: {total_original}")
    print(f"  Total cleaned rows: {total_cleaned}")
    print(f"  Total removed rows: {total_original - total_cleaned}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
