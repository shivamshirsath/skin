import streamlit as st
import numpy as np
from PIL import Image
from model_loader import load_trained_model

st.set_page_config(page_title="AI Skin Disease Detector", layout="centered")

# Load model safely (rebuild + load weights)
@st.cache_resource
def load_model():
    return load_trained_model()

model = load_model()

IMG_SIZE = 224

class_names = [
    'Eczema',
    'Warts Viral Infections',
    'Melanoma',
    'Atopic Dermatitis',
    'Basal Cell Carcinoma (BCC)',
    'Melanocytic Nevi (NV)',
    'Benign Keratosis (BKL)',
    'Psoriasis & Lichen Planus',
    'Seborrheic Keratoses',
    'Tinea Fungal Infections'
]

st.title("🩺 AI-Powered Skin Disease Detector")
st.write("Upload a skin image to get AI prediction.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]

    predicted_index = np.argmax(predictions)
    confidence = float(np.max(predictions))

    st.subheader("🔍 Prediction Result")
    st.success(f"Predicted Disease: {class_names[predicted_index]}")
    st.write(f"Confidence: {confidence*100:.2f}%")

    st.subheader("📊 Top 3 Predictions")
    top_3 = predictions.argsort()[-3:][::-1]

    for idx in top_3:
        st.write(f"{class_names[idx]} — {predictions[idx]*100:.2f}%")
        st.progress(float(predictions[idx]))