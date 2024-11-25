import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# VAE Model definition (Ensure this matches your trained model structure)
class VAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Flatten()
        )
        self.fc_mu = torch.nn.Linear(64 * 16 * 16, 200)
        self.fc_log_var = torch.nn.Linear(64 * 16 * 16, 200)

        # Decoder
        self.fc_dec = torch.nn.Linear(200, 64 * 16 * 16)
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            torch.nn.Sigmoid()
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        z = self.reparameterize(mu, log_var)
        x = self.fc_dec(z).view(-1, 64, 16, 16)
        x = self.decoder(x)
        return mu, log_var, x

# Load the trained model
model_path = r"C:\Users\jasha\Downloads\vae_model.pt"  # Update with your file path
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vae = VAE()
vae.load_state_dict(torch.load(model_path, map_location=DEVICE))
vae.to(DEVICE)
vae.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Streamlit App
st.title("Variational Autoencoder (VAE) - Image Reconstruction")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Transform the image
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)  # Add batch dimension

    # Reconstruct the image using VAE
    with torch.no_grad():
        _, _, reconstructed = vae(input_tensor)

    # Convert reconstructed tensor to image
    reconstructed_image = reconstructed.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # Display reconstructed image
    st.image(reconstructed_image, caption="Reconstructed Image", use_column_width=True)

# Sidebar information
st.sidebar.title("About")
st.sidebar.info("""
This app demonstrates a Variational Autoencoder (VAE) for image reconstruction. Upload an image, and the VAE will reconstruct it based on its learned latent representations.
""")
