# **🐱🐶 Cat vs Dog Classifier (SVM + Streamlit)**

A simple machine learning web app to classify images of cats and dogs using Support Vector Machine (SVM). Users can upload images (jpg/png) or PDFs, and the app predicts whether it’s a cat or dog along with a confidence score.

## 🚀 Features

Upload cat or dog images (jpg / png) or PDFs containing images.

Predict if it’s a Cat 🐱 or Dog 🐶.

Shows confidence score for each prediction.

User-friendly interface with instructions, emojis, and image previews.

Loads the trained SVM model for fast predictions.

## 🛠 How It Works

### Train the Model:

Reads images from Cat and Dog folders.

Preprocesses images (resize to 64x64, grayscale).

Trains a linear SVM classifier and saves the model.

### Run the Web App:

Users upload an image or PDF.

App predicts Cat 🐱 or Dog 🐶 with confidence %.

Displays the uploaded image and prediction results.

## 💻 Technologies Used

Python 3

OpenCV – image processing

scikit-learn – SVM classifier

joblib – saving/loading models

Streamlit – interactive web app

pdf2image – convert PDFs to images
