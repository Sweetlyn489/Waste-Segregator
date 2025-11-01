# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

MODEL_PATH = "biodeg_mobilenetv2.h5"
IMG_SIZE = (224, 224)


@st.cache(allow_output_mutation=True)
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()

st.title("Biodegradable vs Non-Biodegradable Classifier")
st.write("Upload an image of an item and the app will predict whether it's biodegradable.")

uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)
    st.write("")
    st.write("Classifying...")
    img = image.resize(IMG_SIZE)
    arr = np.array(img)/255.0
    arr = np.expand_dims(arr, axis=0)
    proba = model.predict(arr)[0, 0]
    label = "non_biodegradable" if proba >= 0.5 else "biodegradable"
    confidence = proba if proba >= 0.5 else 1-proba
    st.success(f"Prediction: **{label}** (confidence {confidence:.2f})")
    st.write(f"Raw score (sigmoid output): {proba:.4f}")
