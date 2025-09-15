from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import torch
import os

# Check current directory
print("Current working directory:", os.getcwd())

# Load model + processor
model_name = "prithivMLmods/Deep-Fake-Detector-v2-Model"
print("Downloading/loading model...")
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)
print("Model loaded successfully ✅")

# Test with an image
image_path = "data/samples/fake.jpg"
try:
    img = Image.open(image_path)
except Exception as e:
    print(f"Failed to open image {image_path}: {e}")
    exit(1)

inputs = processor(images=img, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

print("Probabilities:", probs)
print("Predicted class:", probs.argmax().item())
