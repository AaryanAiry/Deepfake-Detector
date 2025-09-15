# scripts/downloadtestImages.py

import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

# Initialize Kaggle API
api = KaggleApi()
api.authenticate()

dataset = "manjilkarki/deepfake-and-real-images"

# Destination folders
folders = {
    "real": "data/samples/real",
    "fake": "data/samples/fake"
}
for folder in folders.values():
    os.makedirs(folder, exist_ok=True)

# List all files in the dataset
files = api.dataset_list_files(dataset).files

def download_and_unzip(class_name, count=50):
    # Look specifically in Train/<class_name> folder
    class_files = [f for f in files if f.name.startswith(f"Train/{class_name}/")]
    print(f"Found {len(class_files)} {class_name} images in Train/{class_name} folder.")
    
    for f in class_files[:count]:
        print(f"Downloading {f.name} ...")
        # Download file (usually as zip)
        zip_path = os.path.join(folders[class_name.lower()], os.path.basename(f.name) + ".zip")
        api.dataset_download_file(dataset, f.name, path=folders[class_name.lower()], quiet=False)
        
        # Unzip or move the file
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(folders[class_name.lower()])
            os.remove(zip_path)
            print(f"Extracted {f.name} to {folders[class_name.lower()]}")
        except zipfile.BadZipFile:
            # If the file is not a zip, just rename it
            os.rename(zip_path, os.path.join(folders[class_name.lower()], os.path.basename(f.name)))
            print(f"Moved {f.name} to {folders[class_name.lower()]}")

# Download 50 images from each class
download_and_unzip("Real", 50)
download_and_unzip("Fake", 50)

print("✅ Downloaded 50 Real + 50 Fake images to data/samples/")
