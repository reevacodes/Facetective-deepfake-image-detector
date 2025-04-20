import streamlit as st
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from PIL import Image

# Load best trained model
model_path = os.path.join("best_model.h5")  # Make sure this file exists!
model = load_model(model_path)

IMG_SIZE = 128

# App config
st.set_page_config(page_title="Deepfake Detector", layout="centered")

# App title
st.title("🕵️ Deepfake Image Detector")
st.markdown("Upload an image to check if it's **Real** or **Fake** using a deep learning model.")

# Upload image
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(image, caption="Uploaded Image", width=200)

    # Preprocess the image
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img)
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]  # Remove alpha channel if present
    img_array = img_array.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0]
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    label = "🟢 Real" if predicted_class == 0 else "🔴 Fake"

    with col2:
        st.markdown("### 🧠 Prediction Result")
        st.markdown(f"**Label:** {label}")
        st.markdown(f"**Confidence:** `{confidence:.2f}%`")

else:
    st.info("Please upload an image to begin.")



  