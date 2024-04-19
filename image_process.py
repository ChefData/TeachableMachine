import streamlit as st
from keras.models import load_model as keras_load_model
from PIL import Image, ImageOps
import numpy as np
import os

# Load the Teachable Machine model
def load_teachable_model():
    model_path = 'keras_model.h5'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    # Load the model using Keras
    model = keras_load_model(model_path)
    return model

# Function to preprocess the image
def process_image(image):
    # Convert the image to the required size and format
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image = np.asarray(image)
    image = (image.astype(np.float32) / 255.0)

    # Normalize the image
    image_array = image.reshape(-1, 224, 224, 3)
    return image_array

# Function to make predictions
def predict(model, image1, image2):
    # Process both images
    processed_image1 = process_image(image1)
    processed_image2 = process_image(image2)

    # Make predictions using the model
    predictions = model.predict([processed_image1, processed_image2])

    # Return the prediction results
    return predictions

# Main function
def main():
    st.title("Teachable Machine with Two Inputs")

    # Load the Teachable Machine model
    try:
        model = load_teachable_model()
    except FileNotFoundError as e:
        st.error(str(e))
        return

    # Upload two images
    st.sidebar.title("Upload Images")
    image1 = st.sidebar.file_uploader("Upload First Image", type=['jpg', 'jpeg', 'png'])
    image2 = st.sidebar.file_uploader("Upload Second Image", type=['jpg
