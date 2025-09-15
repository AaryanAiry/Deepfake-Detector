# scripts/evaluateFolder.py

import os
import random
import argparse
from app.models.detector import DeepfakeDetector

def evaluate_split(dataset_root: str, split: str, num_samples: int = 50):
    """
    Evaluate the detector on a dataset split (Train/Test/Validation).
    """

    real_dir = os.path.join(dataset_root, split, "Real")
    fake_dir = os.path.join(dataset_root, split, "Fake")

    if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
        raise FileNotFoundError(f"Could not find {real_dir} or {fake_dir}")

    real_images = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    fake_images = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    # Sample limited number
    real_samples = random.sample(real_images, min(num_samples, len(real_images)))
    fake_samples = random.sample(fake_images, min(num_samples, len(fake_images)))

    detector = DeepfakeDetector()

    correct_real = 0
    correct_fake = 0

    print(f"Evaluating on {split} split with {len(real_samples)} real and {len(fake_samples)} fake images...")

    # Evaluate real images
    for img_path in real_samples:
        result = detector.predict(img_path)
        pred = result["prediction"]
        if pred == "real":
            correct_real += 1
        print(f"{os.path.basename(img_path)}: predicted={pred}, actual=real, confidence={result['confidence']:.2f}")

    # Evaluate fake images
    for img_path in fake_samples:
        result = detector.predict(img_path)
        pred = result["prediction"]
        if pred == "fake":
            correct_fake += 1
        print(f"{os.path.basename(img_path)}: predicted={pred}, actual=fake, confidence={result['confidence']:.2f}")

    total = len(real_samples) + len(fake_samples)
    overall_correct = correct_real + correct_fake
    accuracy = overall_correct / total * 100

    print("\n===== Evaluation Summary =====")
    print(f"Real images: {correct_real}/{len(real_samples)} correct")
    print(f"Fake images: {correct_fake}/{len(fake_samples)} correct")
    print(f"Overall accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="/mnt/MyLinuxSpace/Dataset", help="Path to dataset root")
    parser.add_argument("--split", type=str, choices=["Train", "Test", "Validation"], required=True, help="Which split to evaluate")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of images per class to sample")
    args = parser.parse_args()

    evaluate_split(args.dataset_root, args.split, args.num_samples)
