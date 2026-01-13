import os
import requests
import yaml
from tqdm import tqdm #for progress bar 

def load_config(path=None):
    """
    Load YAML configuration file.
    If path is None, automatically locate 'config/config.yaml' 
    relative to this script.
    """
    if path is None:
        current_script_dir = os.path.dirname(os.path.abspath(__file__))  # Folder of this script
        project_root = os.path.dirname(current_script_dir)               # Go up one level
        path = os.path.join(project_root, 'config', 'config.yaml')       # Path to config.yaml

    with open(path, 'r') as f:
        return yaml.safe_load(f)


    with open(path, 'r') as f:
        return yaml.safe_load(f)

def download_file(url, save_path):
    """Download a files."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure save directory exists
    
    with open(save_path, 'wb') as file, tqdm(
        desc=os.path.basename(save_path),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)



def download_data_pipeline():
    """Main pipeline to download all required data files."""
    config = load_config()  # Load config automatically
    
    # Compute absolute path for data directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_dir_name = config['data_settings']['directory']
    data_dir = os.path.join(project_root, raw_data_dir_name)
    
    base_url = config['data_settings']['base_url']
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    files = [
        'Prix2022S1.csv.gz', 'Prix2022S2.csv.gz', 
        'Prix2023.csv.gz', 'Prix2024.csv.gz', 
        'Stations2024.csv.gz', 'Services2024.csv.gz'
    ]
    
    # Download each file if it doesn't already exist
    for filename in files:
        file_path = os.path.join(data_dir, filename)
        if not os.path.exists(file_path):
            print(f"Downloading {filename}...")
            download_file(base_url + filename, file_path)
        else:
            print(f"{filename} already exists.")

if __name__ == "__main__":
    download_data_pipeline()