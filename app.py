import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# Load the trained autoencoder model
model = tf.keras.models.load_model("model/autoencoder_skin_cancer.h5", compile=False)

# Set threshold (adjust this based on your experiments)
THRESHOLD = 0.1

# Image preprocessing function
def preprocess_image(img):
    img = img.resize((128, 128))
    img = img.convert("RGB")
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

# Function to compute reconstruction loss
def is_anomalous(img_array):
    reconstructed = model.predict(img_array)
    loss = np.mean(np.abs(img_array - reconstructed))
    return loss, loss > THRESHOLD

# Streamlit UI
st.set_page_config(page_title="Skin Cancer Anomaly Detector", layout="centered")
st.title("🔬 Skin Cancer Detection (Autoencoder-based)")
st.write("Upload a skin image or use your webcam to detect possible cancerous anomalies.")

# Upload or Camera Input
st.markdown("### 📤 Upload or 📸 Capture an Image")

upload_method = st.radio("Choose input method:", ("Upload from device", "Use camera"))

image = None
if upload_method == "Upload from device":
    uploaded_file = st.file_uploader("Upload a skin image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)

elif upload_method == "Use camera":
    camera_file = st.camera_input("Take a picture")
    if camera_file:
        image = Image.open(camera_file)

# Inference
if image:
    st.image(image, caption="Input Image", use_column_width=True)
    st.write("🔍 Analyzing...")

    input_image = preprocess_image(image)
    loss, is_anomaly = is_anomalous(input_image)

    if is_anomaly:
        st.markdown("### ✅ **Normal / Non-Cancerous Skin**")
    else:
        st.markdown("### ⚠️ **Potential Skin Cancer Detected**")

    st.write(f"🧪 Reconstruction Loss: `{loss:.4f}` (Threshold: {THRESHOLD})")
    st.info("Note: This tool is for experimental use only. Consult a dermatologist for any concerns.")
