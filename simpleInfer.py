import argparse
import os
import yaml
import subprocess
from huggingface_hub import hf_hub_download

def main():
    parser = argparse.ArgumentParser(description="Download model from HF and run inference using test.py")
    parser.add_argument("--repo_id", type=str, required=True, help="Hugging Face repository ID (e.g., DeepFense/HABLA_EAT_AASIST_NoAug_Seed42)")
    parser.add_argument("--test_parquet", type=str, required=True, help="Path to the test parquet file")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save downloaded files and results. Defaults to the repo name.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name")
    
    args = parser.parse_args()

    # Determine output directory
    if args.output_dir is None:
        # Use the repository name (replacing slash with underscore for local folder safety)
        args.output_dir = args.repo_id.replace("/", "_") + f"_{args.dataset_name}"
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Downloading model and config from {args.repo_id} to {args.output_dir}...")
    try:
        model_path = hf_hub_download(repo_id=args.repo_id, filename="best_model.pth", local_dir=args.output_dir)
        config_path = hf_hub_download(repo_id=args.repo_id, filename="config.yaml", local_dir=args.output_dir)
    except Exception as e:
        print(f"Error downloading files: {e}")
        return

    print("Modifying config for inference...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['output_dir'] = args.output_dir

    # add relevant metrics to the config
    metrics = {
        "EER": {},
        "ACC": {},
        "F1_SCORE": {},
        "EER_CI": {},
    }
    config['training']['metrics'] = metrics

    # Prepare test configuration
    # We reuse the 'val' configuration structure but replace the data source
    if 'data' in config and 'val' in config['data']:
        test_config = config['data']['val'].copy()
    else:
        # Fallback default if val is missing
        print("Warning: 'val' section missing in config. Using default test settings.")
        test_config = {
            'dataset_type': 'StandardDataset',
            'batch_size': 32,
            'shuffle': False,
            'num_workers': 4,
            'base_transform': [{'type': 'pad', 'max_len': 64000, 'pad_type': 'repeat'}]
        }

    # Update with user provided parquet file
    test_config['parquet_files'] = [args.test_parquet]
    
    # Use the filename as the dataset name for clarity in results
    test_config['dataset_names'] = [args.dataset_name]
    
    # Ensure data section exists and add test config
    if 'data' not in config:
        config['data'] = {}
    config['data']['test'] = test_config
    
    # Write the modified config to a new file
    modified_config_path = os.path.join(args.output_dir, "inference_config.yaml")
    with open(modified_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Modified config saved to {modified_config_path}")
    print(f"Running test.py...")
    
    cmd = [
        "python", "test.py",
        "--config", modified_config_path,
        "--checkpoint", model_path
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\nInference completed successfully.")
        print(f"Results should be in: {args.output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"\nInference failed with error code {e.returncode}.")
    except FileNotFoundError:
        print(f"\nError: Could not find python or test.py")

if __name__ == "__main__":
    main()
