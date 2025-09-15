# scripts/download_sample_images.py

import os
import requests

# Folders to store images
folders = {
    "real": "data/samples/real",
    "fake": "data/samples/fake"
}

# Create directories if they don't exist
for folder in folders.values():
    os.makedirs(folder, exist_ok=True)

# Sample image URLs (public domain / free sources)
real_urls = [
    "https://upload.wikimedia.org/wikipedia/commons/3/36/Albert_Einstein_Head.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/9/99/Obama_family_portrait_in_the_Green_Room.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/1/12/Elon_Musk_Royal_Society.jpg"
]

fake_urls = [
    # These are small deepfake examples from public datasets / demo images
    "https://github.com/ondyari/FaceForensics/raw/master/examples/deepfake/images/00001.jpg",
    "https://github.com/ondyari/FaceForensics/raw/master/examples/deepfake/images/00002.jpg",
    "https://github.com/ondyari/FaceForensics/raw/master/examples/deepfake/images/00003.jpg"
]

def download_images(url_list, save_folder):
    for i, url in enumerate(url_list):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                ext = url.split('.')[-1]
                path = os.path.join(save_folder, f"img_{i}.{ext}")
                with open(path, "wb") as f:
                    f.write(response.content)
                print(f"Downloaded {path}")
            else:
                print(f"Failed to download {url}: {response.status_code}")
        except Exception as e:
            print(f"Error downloading {url}: {e}")

# Download real and fake images
download_images(real_urls, folders["real"])
download_images(fake_urls, folders["fake"])

print("✅ Sample images downloaded to data/samples/")
