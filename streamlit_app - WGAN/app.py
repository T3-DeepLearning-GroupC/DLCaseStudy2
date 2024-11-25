import streamlit as st
import torch
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from generator import Generator  # Ensure this path matches your file structure

# Load the pre-trained generator
@st.cache_resource
def load_generator(z_dim=100, channels_img=3, features_g=64, model_path="generator.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator(z_dim, channels_img, features_g).to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    generator.eval()
    return generator, device

# Generate fake images
def generate_images(generator, device, num_images, z_dim):
    noise = torch.randn(num_images, z_dim, 1, 1).to(device)
    with torch.no_grad():
        fake_images = generator(noise)
    fake_images = (fake_images + 1) / 2  # Normalize from [-1, 1] to [0, 1]
    return fake_images

# Streamlit app
st.title("WGAN Image Generator")
st.sidebar.header("Settings")

# Sidebar settings
z_dim = st.sidebar.number_input("Latent Dimension (z_dim)", value=100, min_value=1, max_value=512)
channels_img = st.sidebar.number_input("Output Image Channels", value=3, min_value=1, max_value=3)
features_g = st.sidebar.number_input("Generator Features (features_g)", value=64, min_value=1, max_value=512)
num_images = st.sidebar.slider("Number of Images", min_value=1, max_value=10, value=1)
generator_path = st.sidebar.text_input("Generator Model Path", "generator.pth")

# Load generator
try:
    generator, device = load_generator(z_dim, channels_img, features_g, generator_path)
except Exception as e:
    st.error(f"Failed to load the generator model: {e}")
    st.stop()

if st.sidebar.button("Generate Images"):
    st.write("Generating images...")
    fake_images = generate_images(generator, device, num_images, z_dim)

    # Display images
    st.write("### Generated Images:")
    cols = st.columns(num_images)
    for i, img_tensor in enumerate(fake_images):
        img = img_tensor.permute(1, 2, 0).cpu().numpy()  # CHW to HWC
        img = (img * 255).astype("uint8")  # Convert to uint8 for display
        cols[i].image(img, use_column_width=True)

st.write("Adjust the settings in the sidebar and click **Generate Images** to see the output!")
