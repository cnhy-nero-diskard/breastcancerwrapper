import gradio as gr
import numpy as np
import cv2
import tensorflow as tf

# Load the pre-trained model
model = tf.keras.models.load_model('version1.h5')

# Define the preprocessing function
def preprocess_image(image):
    # Resize the image to 50x50 pixels
    resized_image = cv2.resize(image, (50, 50), interpolation=cv2.INTER_LINEAR)
    # Convert the image to a numpy array and expand dimensions to match the model input shape
    input_image = np.expand_dims(resized_image, axis=0)
    return input_image

# Define the prediction function
def predict_image(image):
    # Preprocess the image
    input_image = preprocess_image(image)
    # Make a prediction
    prediction = model.predict(input_image)
    # Get the predicted class (0 or 1)
    predicted_class = np.argmax(prediction, axis=1)[0]
    # Get the probabilities for both classes
    class_0_prob = prediction[0][0]
    class_1_prob = prediction[0][1]
    # Return the predicted class and probabilities
    return f"Predicted Class: {predicted_class}\nProbability for Class 0: {class_0_prob:.6f}\nProbability for Class 1: {class_1_prob:.6f}"
# Create a Gradio interface
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image( image_mode="RGB"),
    outputs="text",
    title="CNN Model Prediction",
    description="Upload an image to predict its class using a pre-trained CNN model."
)

# Launch the Gradio interface
interface.launch()