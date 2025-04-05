import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("model/skin_cancer_model.h5")

# Function to preprocess the uploaded image
def preprocess_image(img):
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

# Streamlit UI
st.set_page_config(page_title="Skin Cancer Detector", layout="centered")
st.title("🧠 Skin Cancer Detection from Image")
st.write("Upload a skin image and let AI help identify possible skin cancer.")

# File uploader
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("🔍 Analyzing...")
    
    # Preprocess and predict
    input_image = preprocess_image(image)
    prediction = model.predict(input_image)[0][0]

    # Show results
    result = "⚠️ **Potential Skin Cancer Detected**" if prediction > 0.5 else "✅ **Normal / Benign Skin Condition**"
    confidence = float(prediction) if prediction > 0.5 else 1 - float(prediction)

    st.markdown(f"### {result}")
    st.markdown(f"🧪 Confidence: **{confidence * 100:.2f}%**")
    st.info("Note: This is an AI-based early screening tool. Please consult a dermatologist for confirmation.")
