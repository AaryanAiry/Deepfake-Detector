import cv2
import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

# ---- Load your pretrained model ----
def load_model(self, model_name="prithivMLmods/Deep-Fake-Detector-v2-Model", device: str = None):
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
        self.class_names = ["fake","real"]
        
# def load_model():
#     # TODO: replace with your actual model loading function
#     model = torch.load("pretrained_model.pth", map_location="cpu")
#     model.eval()
#     return model

# ---- Preprocessing for each frame ----
transform = transforms.Compose([
    transforms.Resize((128, 128)),   # adjust to your model’s expected size
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def preprocess_frame(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return transform(img).unsqueeze(0)  # add batch dimension

# ---- Predict one frame ----
def predict_frame(model, frame_tensor):
    with torch.no_grad():
        output = model(frame_tensor)
        prob = torch.softmax(output, dim=1).cpu().numpy()[0]
    return prob  # e.g. [real_prob, fake_prob]

# ---- Process video ----
def analyze_video(video_path, frame_skip=15):
    model = load_model()
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    results = []

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            tensor = preprocess_frame(frame)
            probs = predict_frame(model, tensor)
            results.append(probs)

        frame_idx += 1

    cap.release()

    if not results:
        print("No frames processed!")
        return None

    results = np.array(results)  # shape: (N, 2)
    mean_probs = results.mean(axis=0)

    final_label = "Fake" if mean_probs[1] > mean_probs[0] else "Real"
    confidence = max(mean_probs)

    print(f"Final Prediction: {final_label}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Avg Real Score: {mean_probs[0]:.4f}, Avg Fake Score: {mean_probs[1]:.4f}")

    return final_label, confidence, mean_probs
