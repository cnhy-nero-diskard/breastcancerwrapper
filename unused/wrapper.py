import gradio as gr
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the model
model = load_model('version1.h5')
BASE_DIR = "./"

# Preprocessing function
def preprocess_image(image):
    # Resize the image to (50, 50)
    img_resized = cv2.resize(image, (50, 50), interpolation=cv2.INTER_LINEAR)
    # Normalize the image (divide by 255.0)
    img_normalized = img_resized / 255.0
    # Expand dimensions to match the model's input shape (batch dimension)
    img_input = np.expand_dims(img_normalized, axis=0)
    return img_input

# Prediction function
def predict_image(image):
    # Preprocess the image
    processed_image = preprocess_image(image)
    # Make prediction
    prediction = model.predict(processed_image)
    # Assuming binary classification (0 = non-cancerous, 1 = cancerous)
    # Access the first element of the prediction array
    print("Model Prediction:", prediction)
    if prediction[0][0] > 0.5:
        return f"Cancerous (Confidence: {prediction[0][0]:.2f})"
    else:
        return f"Non-Cancerous (Confidence: {1 - prediction[0][0]:.2f})"

# Create the Gradio interface
interface = gr.Interface(
    fn=predict_image,  # Function to call for prediction
    inputs=gr.Image(image_mode="RGB"),  # Input: Image (no shape parameter)
    outputs="text",  # Output: Text label
    title="Breast Cancer Classification Demo",
    description="Upload a breast tissue image to classify it as cancerous or non-cancerous.",
    examples=[
        "path_to_example_image_1.jpg",
        "path_to_example_image_2.jpg",
    ],  # Optional: Add example images for testing
)

# Launch the interface
interface.launch()