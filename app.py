import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import gdown

# Your Google Drive file ID
file_id = '1jjP1s9s9F9K2VVazQMpuKS-GwOBahKqz'
model_path = 'fruit_model.h5'

# Download model if not already downloaded
if not os.path.exists(model_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, model_path, quiet=False)

# Load the model
model = tf.keras.models.load_model(model_path)
class_names = ['jackfruit', 'mango', 'peach', 'pineapple']  # Change if your label order is different

# Streamlit UI
st.title("🍓 Fruit Classifier App")
st.markdown("Upload an image of a fruit and get the prediction instantly!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = img.resize((180, 180))  # match your model input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_label = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    st.markdown(f"### 🧠 Prediction: `{predicted_label}`")
    st.markdown(f"### ✅ Confidence: `{confidence:.2f}%`")
