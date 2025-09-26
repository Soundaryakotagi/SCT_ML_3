# app.py
import streamlit as st
import cv2
import numpy as np
import joblib
from pdf2image import convert_from_bytes
import os

# Load model
model_path = r"C:\Users\nanda\OneDrive\Documents\Nanda's ML Tasks\SVM\PetImages\cat_dog_svm_model.pkl"
clf = joblib.load(model_path)

# Streamlit UI
st.title("🐱 Cat vs 🐶 Dog Classifier")
uploaded_file = st.file_uploader("Upload an image or PDF", type=["jpg", "png", "pdf"])

def classify_image(img):
    img = cv2.resize(img, (64, 64))
    img = img.flatten().reshape(1, -1)
    prediction = clf.predict(img)[0]
    return "Cat 🐱" if prediction == 0 else "Dog 🐶"

def process_pdf(pdf_file):
    images = convert_from_bytes(pdf_file.read())
    return [cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY) for img in images]

if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        st.info("PDF uploaded. Extracting images...")
        images = process_pdf(uploaded_file)
        for i, img in enumerate(images):
            st.image(img, caption=f"Page {i+1}", use_column_width=True)
            st.write("Prediction:", classify_image(img))
    else:
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        st.write("Prediction:", classify_image(img))
