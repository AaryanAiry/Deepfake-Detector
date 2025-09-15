# app/services/inference.py

import os
from app.models.detector import DeepfakeDetector


class InferenceService:
    def __init__(self):
        """
        Initialize the inference service with the deepfake detector model.
        """
        self.detector = DeepfakeDetector()

    def run_inference(self, image_path: str) -> dict:
        """
        Run deepfake detection on a given image file.

        Args:
            image_path (str): Path to the image file.

        Returns:
            dict: { "prediction": "real/fake", "confidence": float }
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        result = self.detector.predict(image_path)
        return result


if __name__ == "__main__":
    # Quick test
    service = InferenceService()
    test_img = "data/samples/fake.jpg"  
    output = service.run_inference(test_img)
    print(output)
