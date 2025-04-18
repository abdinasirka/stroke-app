import gdown
import os
import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# ✅ Set model file path
model_path = "stroke_classification_model.h5"


file_id = "1fA5eHtk9w-wXX3RgjM5iwy1b4P3OSXxd"
url = f"https://drive.google.com/uc?id={file_id}"
output = "stroke_classification_model.h5"

# Download the model file if it doesn't already exist
import os
if not os.path.exists(output):
    gdown.download(url, output, quiet=False)


# ✅ Load the trained model
model = load_model(model_path)

# Image dimensions
IMG_SIZE = (224, 224)

# Labels
labels = ['Normal', 'Stroke']

# UI
st.title("🧠 Stroke Detection from Brain Scans")
st.write("Upload a brain scan image and the model will classify it as Normal or Stroke.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, channels="BGR", caption="Uploaded Image", use_container_width=True)

    # Preprocess
    image_resized = cv2.resize(image, IMG_SIZE)
    image_array = img_to_array(image_resized)
    image_array = np.expand_dims(image_array, axis=0) / 255.0

    # Predict
    prediction = model.predict(image_array)[0]
    class_idx = np.argmax(prediction)
    confidence = prediction[class_idx]

    # Show result
    st.write(f"### Prediction: **{labels[class_idx]}**")
    st.write(f"Confidence: {confidence:.2f}")
