import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load model
model = load_model("CNN_Model/emotion_model_bc.h5")

# Class labels
class_names = ["Disgust", "Happy"]

st.title("Emotion Classifier - Happy vs Disgust")

uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    image = image.resize((48, 48))
    image_array = np.array(image) / 255.0
    image_array = image_array.reshape(1, 48, 48, 1)

    # Predict
    pred = model.predict(image_array)[0][0]  # sigmoid output
    confidence = max(pred, 1 - pred)

    if pred >= 0.60:
        predicted_class = 1  # Happy
        label = class_names[predicted_class]
    else:
        predicted_class = 0  # Disgust
        label = class_names[predicted_class]
    

    st.write(f"**Prediction:** {label}")
   
