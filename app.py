# app.py
import streamlit as st
from PIL import Image
from model_utils import load_model, preprocess_image

st.set_page_config(page_title="Digit Recognition", layout="centered")
st.title("🧠 Handwritten Digit Recognition")

# Load pre-trained model
model = load_model()

# Upload image
uploaded_file = st.file_uploader("📤 Upload a digit image (28x28, white digit on black background)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", width=150)
    with st.spinner("Predicting..."):
        img = Image.open(uploaded_file)
        img_array, processed_img = preprocess_image(img)
        prediction = model.predict(img_array)[0]
        st.image(processed_img, caption=f"🧾 Predicted Digit: {prediction}", width=150)
        st.success(f"✅ Prediction: {prediction}")


