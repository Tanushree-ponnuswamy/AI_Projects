# 🧠 Smart Wearable for Skin Cancer Detection using Autoencoder

This project is a prototype for a **smart wearable device** designed to **detect skin cancer using AI**, specifically using a **one-class classification approach with an autoencoder**. The goal is to enable early-stage detection of cancerous skin lesions by identifying anomalies in skin images in real-time.

## 📌 Problem Statement

Traditional image classification models struggle to identify unknown or non-cancerous cases if they haven't been trained on them. This wearable project tackles that problem by:
- Training only on **cancerous skin images**
- Using an **autoencoder** to detect **non-cancerous images as anomalies**

This approach ensures better performance in **real-world environments**, where new, unseen, or non-cancerous conditions may occur.

---

## 🚀 Project Highlights

- ✅ **Autoencoder-based anomaly detection** for one-class classification
- ✅ Real-time analysis using **Streamlit app**
- ✅ Supports **smart wearable integration** (e.g. ESP32 + camera module)
- ✅ Lightweight model (ideal for mobile or embedded devices)
- ✅ Modular codebase: Easily adaptable for different types of cancers or conditions

---

## 🧠 How It Works

- The autoencoder is trained **only on cancerous images**
- At inference time, the model **reconstructs** the input image
- If the **reconstruction loss is high**, it is flagged as **non-cancerous (anomaly)**
- If the **reconstruction is good**, it is **likely cancerous**

---

## 🗂 Folder Structure

smart_wearable/ │ ├── data/ │ └── train/ │ ├── image/ ← Contains cancerous skin images │ └── dummy/ ← Contains one or few dummy non-cancerous images │ ├── model/ │ └── autoencoder_skin_cancer.h5 ← Saved model │ ├── train_model.py ← Code to train autoencoder ├── app.py ← Streamlit app for detection └── README.md


---

## 📱 Smart Wearable Tech Stack (Prototype)

- **Hardware**: ESP32 / Raspberry Pi + Camera Module + Optical/UV Sensor
- **AI Model**: Autoencoder using TensorFlow / Keras
- **App Interface**: Streamlit for demo, TFLite for mobile integration
- **Connectivity**: Bluetooth 5.0 / WiFi for real-time data transmission

---

## 📦 Installation

```bash
pip install -r requirements.txt

then run the Streamlit app
streamlit run app.py

📌 Future Work
Integrate real-time detection with a wearable band

Extend the model to other types of cancer

Improve accuracy with more diverse data

Add mobile app support (Flutter/Kotlin + TFLite)

🧬 Disclaimer
This project is a research prototype for educational and proof-of-concept purposes. It is not intended for clinical use without validation from medical professionals.

---

Would you like me to auto-generate a `requirements.txt` or a license file too?
