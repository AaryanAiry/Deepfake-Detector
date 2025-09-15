from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from app.models.detector import DeepfakeDetector
import shutil
from pathlib import Path
import mimetypes

# import your video analyzer
from app.models.analyze_video import analyze_video  

app = FastAPI()
detector = DeepfakeDetector()

# Allow Streamlit frontend to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

UPLOAD_FOLDER = Path("data/uploads")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    file_path = UPLOAD_FOLDER / file.filename
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    mime_type, _ = mimetypes.guess_type(file.filename)

    # Image prediction
    if mime_type and mime_type.startswith("image"):
        result = detector.predict(str(file_path))
        return {
            "type": "image",
            "prediction": result["prediction"],
            "confidence": result["confidence"],
        }

    # Video prediction
    elif mime_type and mime_type.startswith("video"):
        label, confidence, mean_probs, frames_analyzed = analyze_video(str(file_path), detector)
        return {
            "type": "video",
            "prediction": label,
            "confidence": confidence,
            "real_score": float(mean_probs["real"]),
            "fake_score": float(mean_probs["fake"]),
            "frames_analyzed": frames_analyzed,
        }

    else:
        return {"error": "Unsupported file type"}
