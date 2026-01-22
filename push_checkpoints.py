import os
from huggingface_hub import HfApi

# Configuration
TOKEN = ""
FILES_TO_UPLOAD = ["best_model.pth", "config.yaml"]

def main():
    api = HfApi(token=TOKEN)
    
    ROOT_DIR = "/netscratch/yelkheir/DeepFense/DeepFense/outputs/batch5_CtrSVDD"

    # Verify root directory exists
    if not os.path.exists(ROOT_DIR):
        print(f"Error: Directory {ROOT_DIR} does not exist.")
        return

    # Get user info to construct repo_id correctly (username/repo_name)
    # user_info = api.whoami()
    # username = user_info['name']
    # print(f"Logged in as: {username}")

    # Iterate over all items in the root directory
    for folder_name in sorted(os.listdir(ROOT_DIR)):
        folder_path = os.path.join(ROOT_DIR, folder_name)
        
        # Skip if it's not a directory
        if not os.path.isdir(folder_path):
            continue

        print(f"\nProcessing folder: {folder_name}")

        # Check if any of the target files exist in this folder
        files_found = []
        for filename in FILES_TO_UPLOAD:
            if os.path.exists(os.path.join(folder_path, filename)):
                files_found.append(filename)
        
        if not files_found:
            print(f"  No relevant files ({', '.join(FILES_TO_UPLOAD)}) found. Skipping.")
            continue

        # Construct Repository Name
        # Using the folder name as the repo name. 
        # Note: HF repo names allow alphanumeric, '.', '-', '_'.
        repo_name = '_'.join(folder_name.split("_")[:5])
        # repo_name = "ASV19_" + repo_name
        repo_name = repo_name.replace("Hable", "HABLA")
        repo_name = repo_name.replace("seed", "Seed")
        repo_name = repo_name.replace("eat", "EAT")
        repo_name = repo_name.replace("hubert", "Hubert")
        repo_name = repo_name.replace("wavlm", "WavLM")
        repo_name = repo_name.replace("wav2vec2", "Wav2Vec2")

        print(f"Repo name: {repo_name}")
        repo_id = f"DeepFense/{repo_name}"
        
        try:
            # Create repository (private by default to avoid accidental leaks, change private=False if needed)
            print(f"  Creating/Checking repo: {repo_id}")
            api.create_repo(repo_id=repo_id, exist_ok=True, private=False)
            
            # Upload files
            for filename in files_found:
                file_path = os.path.join(folder_path, filename)
                print(f"  Uploading {filename}...")
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=filename,
                    repo_id=repo_id,
                    repo_type="model"
                )
            print(f"  Successfully uploaded to https://huggingface.co/{repo_id}")
            
        except Exception as e:
            print(f"  Error processing {folder_name}: {e}")


    ROOT_DIR = "/netscratch/yelkheir/DeepFense/DeepFense/outputs/batch5_EnvSSD"
    if not os.path.exists(ROOT_DIR):
        print(f"Error: Directory {ROOT_DIR} does not exist.")
        return

    # Get user info to construct repo_id correctly (username/repo_name)
    # user_info = api.whoami()
    # username = user_info['name']
    # print(f"Logged in as: {username}")

    # Iterate over all items in the root directory
    for folder_name in sorted(os.listdir(ROOT_DIR)):
        folder_path = os.path.join(ROOT_DIR, folder_name)
        
        # Skip if it's not a directory
        if not os.path.isdir(folder_path):
            continue

        print(f"\nProcessing folder: {folder_name}")

        # Check if any of the target files exist in this folder
        files_found = []
        for filename in FILES_TO_UPLOAD:
            if os.path.exists(os.path.join(folder_path, filename)):
                files_found.append(filename)
        
        if not files_found:
            print(f"  No relevant files ({', '.join(FILES_TO_UPLOAD)}) found. Skipping.")
            continue

        # Construct Repository Name
        # Using the folder name as the repo name. 
        # Note: HF repo names allow alphanumeric, '.', '-', '_'.
        repo_name = '_'.join(folder_name.split("_")[:5])
        # repo_name = "ASV19_" + repo_name
        repo_name = repo_name.replace("Hable", "HABLA")
        repo_name = repo_name.replace("seed", "Seed")
        repo_name = repo_name.replace("eat", "EAT")
        repo_name = repo_name.replace("hubert", "Hubert")
        repo_name = repo_name.replace("wavlm", "WavLM")
        repo_name = repo_name.replace("wav2vec2", "Wav2Vec2")

        print(f"Repo name: {repo_name}")
        repo_id = f"DeepFense/{repo_name}"
        
        try:
            # Create repository (private by default to avoid accidental leaks, change private=False if needed)
            print(f"  Creating/Checking repo: {repo_id}")
            api.create_repo(repo_id=repo_id, exist_ok=True, private=False)
            
            # Upload files
            for filename in files_found:
                file_path = os.path.join(folder_path, filename)
                print(f"  Uploading {filename}...")
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=filename,
                    repo_id=repo_id,
                    repo_type="model"
                )
            print(f"  Successfully uploaded to https://huggingface.co/{repo_id}")
            
        except Exception as e:
            print(f"  Error processing {folder_name}: {e}")


    ROOT_DIR = "/netscratch/yelkheir/DeepFense/DeepFense/outputs/batch6_ASV5"
    if not os.path.exists(ROOT_DIR):
        print(f"Error: Directory {ROOT_DIR} does not exist.")
        return

    # Get user info to construct repo_id correctly (username/repo_name)
    # user_info = api.whoami()
    # username = user_info['name']
    # print(f"Logged in as: {username}")

    # Iterate over all items in the root directory
    for folder_name in sorted(os.listdir(ROOT_DIR)):
        folder_path = os.path.join(ROOT_DIR, folder_name)
        
        # Skip if it's not a directory
        if not os.path.isdir(folder_path):
            continue

        print(f"\nProcessing folder: {folder_name}")

        # Check if any of the target files exist in this folder
        files_found = []
        for filename in FILES_TO_UPLOAD:
            if os.path.exists(os.path.join(folder_path, filename)):
                files_found.append(filename)
        
        if not files_found:
            print(f"  No relevant files ({', '.join(FILES_TO_UPLOAD)}) found. Skipping.")
            continue

        # Construct Repository Name
        # Using the folder name as the repo name. 
        # Note: HF repo names allow alphanumeric, '.', '-', '_'.
        repo_name = '_'.join(folder_name.split("_")[:5])
        # repo_name = "ASV19_" + repo_name
        repo_name = repo_name.replace("Hable", "HABLA")
        repo_name = repo_name.replace("seed", "Seed")
        repo_name = repo_name.replace("eat", "EAT")
        repo_name = repo_name.replace("hubert", "Hubert")
        repo_name = repo_name.replace("wavlm", "WavLM")
        repo_name = repo_name.replace("wav2vec2", "Wav2Vec2")

        print(f"Repo name: {repo_name}")
        repo_id = f"DeepFense/{repo_name}"
        
        try:
            # Create repository (private by default to avoid accidental leaks, change private=False if needed)
            print(f"  Creating/Checking repo: {repo_id}")
            api.create_repo(repo_id=repo_id, exist_ok=True, private=False)
            
            # Upload files
            for filename in files_found:
                file_path = os.path.join(folder_path, filename)
                print(f"  Uploading {filename}...")
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=filename,
                    repo_id=repo_id,
                    repo_type="model"
                )
            print(f"  Successfully uploaded to https://huggingface.co/{repo_id}")
            
        except Exception as e:
            print(f"  Error processing {folder_name}: {e}")


if __name__ == "__main__":
    main()
