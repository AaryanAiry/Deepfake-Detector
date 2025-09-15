# frontend/streamlitApp.py
import streamlit as st
import requests
from PIL import Image

st.title("Deepfake Detector")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)  # updated param

    if st.button("Check Deepfake"):
        try:
            # Send image to FastAPI backend
            files = {"file": (uploaded_file.name, uploaded_file.getbuffer(), uploaded_file.type)}
            response = requests.post("http://127.0.0.1:8000/predict/", files=files)

            if response.status_code == 200:
                data = response.json()
                st.success(f"Prediction: {data['prediction'].capitalize()}")
                st.info(f"Confidence: {data['confidence']:.2f}")
            else:
                st.error(f"Prediction failed! Status code: {response.status_code}")
        except Exception as e:
            st.error(f"Prediction failed! Error: {e}")
