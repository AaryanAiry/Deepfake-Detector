# app/models/detector.py

from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import torch
import torch.nn.functional as F

class DeepfakeDetector:
    def __init__(self, model_name="prithivMLmods/Deep-Fake-Detector-v2-Model", device: str = None):
        """
        Initialize the deepfake detector with the Hugging Face pretrained model.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load model + processor
        print("Loading Hugging Face model...")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print("Model loaded ✅")

        # Labels (assumes 0=real, 1=fake)
        self.class_names = ["real","fake"]

    def predict(self, image_path: str):
        """
        Run inference on a single image and return prediction + confidence.
        """
        img = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1).cpu()
            pred_idx = probs.argmax(-1).item()

        return {
            "prediction": self.class_names[pred_idx],
            "confidence": float(probs[0][pred_idx])
        }


if __name__ == "__main__":
    # Quick test
    # detector = DeepfakeDetector()
    # print(detector.predict("data/samples/real.jpg"))
    pass
