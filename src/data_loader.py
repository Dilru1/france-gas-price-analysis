import os
import requests
import yaml
from tqdm import tqdm

def load_config(path=None):
    # If no specific path is provided, find it automatically
    if path is None:
        # 1. Get the folder where THIS script (data_loader.py) lives (i.e., 'src/')
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 2. Go up one level to the Project Root
        project_root = os.path.dirname(current_script_dir)
        
        # 3. Build the absolute path to config.yaml
        path = os.path.join(project_root, 'config', 'config.yaml')

    with open(path, 'r') as f:
        return yaml.safe_load(f)

def download_file(url, save_path):
    """Downloads a file with a progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    # Ensure the directory for the file exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
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
    # No arguments needed; it will find the config automatically now
    config = load_config() 
    
    # We also need to fix the data directory path
    # If config says 'gas_data', we want it to be absolute path: 'project_root/gas_data'
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    raw_data_dir_name = config['data_settings']['directory']
    # Construct absolute path for data to avoid CWD issues
    data_dir = os.path.join(project_root, raw_data_dir_name)
    
    base_url = config['data_settings']['base_url']
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    files = [
        'Prix2022S1.csv.gz', 'Prix2022S2.csv.gz', 
        'Prix2023.csv.gz', 'Prix2024.csv.gz', 
        'Stations2024.csv.gz', 'Services2024.csv.gz'
    ]
    
    for filename in files:
        file_path = os.path.join(data_dir, filename)
        if not os.path.exists(file_path):
            print(f"Downloading {filename}...")
            download_file(base_url + filename, file_path)
        else:
            print(f"{filename} already exists.")

if __name__ == "__main__":
    download_data_pipeline()