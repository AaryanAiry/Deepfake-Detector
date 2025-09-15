import streamlit as st
import requests
from PIL import Image
import mimetypes

st.title("Deepfake Detector")

uploaded_file = st.file_uploader(
    "Choose an image or video...", 
    type=["jpg", "png", "jpeg", "mp4", "avi", "mov", "mkv"]
)

if uploaded_file is not None:
    mime_type, _ = mimetypes.guess_type(uploaded_file.name)

    if mime_type and mime_type.startswith("image"):
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width="stretch")  # new param

    elif mime_type and mime_type.startswith("video"):
        st.video(uploaded_file)

    if st.button("Check Deepfake"):
        try:
            files = {"file": (uploaded_file.name, uploaded_file.getbuffer(), mime_type)}
            response = requests.post("http://127.0.0.1:8000/predict/", files=files)

            if response.status_code == 200:
                data = response.json()

                if data.get("type") == "image":
                    st.success(f"Prediction: {data['prediction'].capitalize()}")
                    st.info(f"Confidence: {data['confidence']:.2f}")

                elif data.get("type") == "video":
                    st.success(f"Final Prediction: {data['prediction'].capitalize()}")
                    st.info(f"Confidence: {data['confidence']:.2f}")
                    st.write(f"Avg Real Score: {data['real_score']:.2f}")
                    st.write(f"Avg Fake Score: {data['fake_score']:.2f}")
                    st.write(f"Frames Analyzed: {data['frames_analyzed']}")

            else:
                st.error(f"Prediction failed! Status code: {response.status_code}")

        except Exception as e:
            st.error(f"Prediction failed! Error: {e}")
