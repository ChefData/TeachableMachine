import streamlit as st
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np

# Load the Teachable Machine model
def load_model():
    # Load the model from Keras
    model = tensorflow.keras.models.load_model('keras_model.h5')
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
    model = load_model()

    # Upload two images
    st.sidebar.title("Upload Images")
    image1 = st.sidebar.file_uploader("Upload First Image", type=['jpg', 'jpeg', 'png'])
    image2 = st.sidebar.file_uploader("Upload Second Image", type=['jpg', 'jpeg', 'png'])

    if image1 is not None and image2 is not None:
        # Display the uploaded images
        st.sidebar.image(image1, caption='First Image', use_column_width=True)
        st.sidebar.image(image2, caption='Second Image', use_column_width=True)

        # Make predictions when the user clicks the button
        if st.sidebar.button("Predict"):
            predictions = predict(model, Image.open(image1), Image.open(image2))
            st.write("Prediction:", predictions)

# Run the main function
if __name__ == "__main__":
    main()
