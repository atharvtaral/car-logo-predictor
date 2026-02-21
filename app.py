import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# --- 1. CSS for Decoration ---
st.set_page_config(page_title="Car Logo AI", page_icon="🚗", layout="centered")

st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #1e1e2f 0%, #121212 100%);
        color: white;
    }

    /* Header styling */
    h1 {
        color: #00d4ff;
        text-align: center;
        font-family: 'Trebuchet MS', sans-serif;
        text-shadow: 2px 2px 10px rgba(0,212,255,0.3);
    }

    /* Upload box styling */
    .stFileUploader {
        border: 2px dashed #00d4ff;
        border-radius: 15px;
        padding: 10px;
        background: rgba(255, 255, 255, 0.05);
    }

    /* Custom Success & Info Boxes */
    .stAlert {
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }

    /* Image display styling */
    img {
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    </style>
    """, unsafe_allow_html=True)


# --- 2. Load Model & Classes ---
@st.cache_resource
def load_my_model():
    # Use your saved model name here
    model = tf.keras.models.load_model(r"C:\Users\Shree\OneDrive\Documents\Gen Ai Projects\Classification Projects\Car logo\upload\car_logo_model_91.h5")
    return model


model = load_my_model()

with open('classes.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# --- 3. UI Layout ---
st.markdown("<h1>🚗 Intelligent Car Logo Classifier</h1>", unsafe_allow_html=True)
st.write(
    "<p style='text-align: center; color: #bbb;'>Experience high-accuracy brand recognition powered by Deep Learning</p>",
    unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose a car logo image...", type=["jpg", "jpeg", "png", "jfif", "webp", "bmp"])

if uploaded_file is not None:
    # Processing
    img = Image.open(uploaded_file)
    st.image(img, caption='Analyzing this image...', use_container_width=True)

    with st.spinner('Artificial Intelligence is thinking...'):
        # Prepare image for model
        img_input = img.resize((128, 128))
        img_array = image.img_to_array(img_input)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        predictions = model.predict(img_array)
        score = np.argmax(predictions)
        confidence = np.max(predictions) * 100

        # Result UI
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"### Brand: {class_names[score]}")
        with col2:
            st.info(f"### Confidence: {confidence:.2f}%")

        if confidence < 60:
            st.warning("Low confidence! Please try a clearer image of the logo.")