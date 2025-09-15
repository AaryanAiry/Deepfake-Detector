# scripts/evaluate_model.py

import os
from app.models.detector import DeepfakeDetector

# Initialize detector
detector = DeepfakeDetector()

# Paths
real_dir = "data/samples/real"
fake_dir = "data/samples/fake"

def evaluate_folder(folder_path, true_label):
    images = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".png"))]
    correct = 0
    total = len(images)

    for img_name in images:
        img_path = os.path.join(folder_path, img_name)
        result = detector.predict(img_path)
        pred_label = result["prediction"]
        confidence = result["confidence"]

        print(f"{img_name}: predicted={pred_label}, confidence={confidence:.2f}, actual={true_label}")
        if pred_label.lower() == true_label.lower():
            correct += 1

    return correct, total

# Evaluate Real images
real_correct, real_total = evaluate_folder(real_dir, "real")
# Evaluate Fake images
fake_correct, fake_total = evaluate_folder(fake_dir, "fake")

# Overall accuracy
total_correct = real_correct + fake_correct
total_images = real_total + fake_total
accuracy = total_correct / total_images * 100

print("\n===== Evaluation Summary =====")
print(f"Real images: {real_correct}/{real_total} correct")
print(f"Fake images: {fake_correct}/{fake_total} correct")
print(f"Overall accuracy: {accuracy:.2f}%")
