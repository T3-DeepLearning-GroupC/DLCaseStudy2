import streamlit as st
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
import matplotlib.pyplot as plt

# Define noise dimension globally
noise_dim = 100

# Load the trained generator model
@st.cache(allow_output_mutation=True)
def load_generator():
    num_classes = 4
    generator = tf.keras.models.load_model('cGan.h5')
    return generator

generator = load_generator()

# Function to generate images based on a specific class
def generate_images_by_class(generator, class_label, num_images=10):
    noise = np.random.normal(0, 1, (num_images, noise_dim)).astype(np.float32)
    labels = np.full((num_images,), class_label)  # Create an array with the specified class label
    labels = to_categorical(labels, num_classes=4).astype(np.float32)
    
    generated_images = generator.predict(tf.concat([noise, labels], axis=1))
    generated_images = (generated_images + 1) / 2.0  # Denormalize to [0, 1]
    generated_images = tf.reshape(generated_images, (num_images, 224, 224, 3))  # Adjust for RGB image size
    
    return generated_images

# Streamlit app
st.title("cGAN Image Generator")
st.write("Generate images based on selected class labels using a Conditional GAN.")

# Class selection
class_label = st.selectbox("Select a class label:", ["drink", "food", "inside", "outside"])
class_mapping = {"drink": 0, "food": 1, "inside": 2, "outside": 3}
class_index = class_mapping[class_label]

# Number of images to generate
num_images = st.slider("Number of images to generate:", 1, 10, 5)

# Generate images button
if st.button("Generate Images"):
    generated_images = generate_images_by_class(generator, class_index, num_images)
    
    # Display generated images
    st.write(f"Generated images for class: {class_label}")
    fig, axes = plt.subplots(1, num_images, figsize=(15, 10))
    for i in range(num_images):
        axes[i].imshow(generated_images[i])
        axes[i].axis('off')
    st.pyplot(fig)