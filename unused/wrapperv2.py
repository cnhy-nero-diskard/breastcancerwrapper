import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import gradio as gr

# Load the trained CNN model
MODEL_PATH = 'version1.h5'  # Replace with the path to your model
model = load_model(MODEL_PATH)

# Define class labels
CLASS_LABELS = ['Non-Cancerous', 'Cancerous']

# Preprocessing function
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert if required
    image = cv2.resize(image, (50, 50), interpolation=cv2.INTER_LINEAR)
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Prediction function
def predict(image):
    """
    Predict the class of the uploaded image.
    """
    try:
        # Preprocess the input image
        preprocessed_image = preprocess_image(image)
        # Perform prediction
        prediction = model.predict(preprocessed_image)
        print(f"Prediction probabilities: {prediction}")
        # Get the predicted class
        predicted_class = CLASS_LABELS[np.argmax(prediction)]
        # Get confidence score
        confidence = np.max(prediction)
        return f"Prediction: {predicted_class}\nConfidence: {confidence:.2f}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Gradio Interface
interface = gr.Interface(
    fn=predict, 
    inputs=gr.Image(image_mode="RGB"), 
    outputs="text",
    title="Breast Cancer Classification",
    description="Drag and drop an image to classify it as Non-Cancerous or Cancerous."
)

# Launch the interface
if __name__ == "__main__":
    interface.launch()
