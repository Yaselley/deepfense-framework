import pandas as pd
import argparse
from pathlib import Path

def process_ctrsvdd_dataset(data_root, meta_root):
    """
    Process CtrSVDD dataset by reading test.txt and scanning audio files.
    The dataset contains:
    - test.txt: metadata file with columns: origin source, singer id, filename, placeholder, attack iD, label
    - test_set/ dir: contains .flac audio files
    - Labels: "bonafide" or "deepfake" (mapped to "spoof")

    https://r9y9.github.io/projects/ctrsvdd/
    https://github.com/SVDDChallenge/CtrSVDD2024_Baseline 

    Args:
        data_root: root directory containing the uncompressed CtrSVDD dataset
        meta_root: root directory containing the metadata files
    
    Returns:
        List of dicts containing ID, path, label, and dataset_name
    """
    data_root = Path(data_root)
    meta_root = Path(meta_root)
    if not data_root.exists():
        print(f"Error: Data root directory not found: {data_root}")
        return []
    
    test_txt = meta_root / "test.txt"
    test_set_dir = data_root / "test_set"
    
    if not test_txt.exists():
        print(f"Error: test.txt not found: {test_txt}")
        return []
    
    if not test_set_dir.exists():
        print(f"Error: test_set directory not found: {test_set_dir}")
        return []
    
    all_data = []
    
    with open(test_txt, 'r') as f:
        for line in f:
            line = line.strip()
            
            # origin source, singer id, filename, placeholder, attack iD, label
            parts = line.split()
            assert len(parts) == 6, f"Expected 6 parts, got {len(parts)}"
            
            source = parts[0]
            singer_id = parts[1]
            filename = parts[2]
            placeholder = parts[3]
            attack_id = parts[4]
            label_str = parts[-1]  
            
            # Map "deepfake" to "spoof", keep "bonafide" as is
            if label_str == "deepfake":
                label = "spoof"
            elif label_str == "bonafide":
                label = "bonafide"
            
            flac_file = test_set_dir / f"{filename}.flac"
            
            all_data.append({
                "ID": filename,
                "path": str(flac_file),
                "label": label,
                "dataset_name": "CtrSVDD"
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
    print(f"Writing parquet files to {output_dir}") 
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data_root = Path(args.data_root)
    meta_root = Path(args.meta_root) if args.meta_root is not None else data_root
    all_data = process_ctrsvdd_dataset(data_root, meta_root)
    
    if all_data:
        df = pd.DataFrame(all_data)
        output_path = output_dir / "ctrsvdd.parquet"
        df.to_parquet(output_path)
        print(f"\nSaved {len(df)} rows to {output_path}")
        print(f"Label distribution:")
        print(df['label'].value_counts())
    else:
        print("No data to save!")

if __name__ == "__main__":
    main()
