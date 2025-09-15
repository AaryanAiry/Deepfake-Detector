import cv2
import numpy as np
from pathlib import Path
from PIL import Image

def analyze_video(video_path, detector, frame_skip=2, max_frames=500):
    """
    Analyze a video by sampling frames and passing them through the deepfake detector.
    Args:
        video_path: path to video file
        detector: DeepfakeDetector instance (already loaded in main.py)
        frame_skip: number of frames to skip
        max_frames: max number of frames to process
    Returns:
        final_label: "real" or "fake"
        confidence: float
        mean_probs: dict with real/fake avg scores
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frame_count = 0
    results = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip == 0:
            # Save frame temporarily
            from tempfile import NamedTemporaryFile
            temp_file = NamedTemporaryFile(suffix=".jpg", delete=False)
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img.save(temp_file.name)

            results.append(detector.predict(temp_file.name))
            temp_file.close()

            if len(results) >= max_frames:
                break
        frame_count += 1

    cap.release()

    if not results:
        return "unknown", 0.0, {"real": 0.0, "fake": 0.0}

    # Separate confidences by class
    real_conf = [r["confidence"] for r in results if r["prediction"] == "real"]
    fake_conf = [r["confidence"] for r in results if r["prediction"] == "fake"]

    avg_real = sum(real_conf) / len(real_conf) if real_conf else 0
    avg_fake = sum(fake_conf) / len(fake_conf) if fake_conf else 0

    final_label = "real" if avg_real >= avg_fake else "fake"
    confidence = max(avg_real, avg_fake)

    return final_label, confidence, {"real": avg_real, "fake": avg_fake}
