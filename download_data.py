import os
import shutil
import sys
import yaml

def main():
    print("Starting Kaggle dataset download...")
    
    # Load config to get output path
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        print(f"Error: Config file {config_path} not found.")
        sys.exit(1)
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    raw_path = config["data"]["raw_path"]
    data_dir = os.path.dirname(raw_path)
    
    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)
    
    # Check if file already exists
    if os.path.exists(raw_path):
        print(f"Dataset already exists at {raw_path}. Skipping download.")
        return

    try:
        import kagglehub
    except ImportError:
        print("Error: 'kagglehub' package is not installed. Please install dependencies first:")
        print("pip install -r requirements.txt")
        sys.exit(1)
        
    try:
        print("Downloading 'mlg-ulb/creditcardfraud' via kagglehub...")
        # kagglehub.dataset_download returns the path to the downloaded folder
        download_path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
        print(f"Downloaded folder: {download_path}")
        
        # Look for creditcard.csv in the downloaded files
        csv_filename = "creditcard.csv"
        src_file = os.path.join(download_path, csv_filename)
        
        if not os.path.exists(src_file):
            # Try searching recursively for creditcard.csv
            found = False
            for root, dirs, files in os.walk(download_path):
                if csv_filename in files:
                    src_file = os.path.join(root, csv_filename)
                    found = True
                    break
            if not found:
                raise FileNotFoundError(f"Could not find {csv_filename} inside the downloaded files.")
                
        # Copy to local data directory
        print(f"Copying {src_file} to {raw_path}...")
        shutil.copy(src_file, raw_path)
        print("Dataset successfully set up!")
        
    except Exception as e:
        print(f"An error occurred while downloading/copying the dataset: {e}")
        print("Make sure you have an active internet connection and that kagglehub is set up correctly.")
        sys.exit(1)

if __name__ == "__main__":
    main()
