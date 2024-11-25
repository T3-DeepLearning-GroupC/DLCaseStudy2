import matplotlib.pyplot as plt
import torch

def plot_generated_images(data_loader, model, device, modeltype='VAE', num_images=8):
    """
    Generate and plot images using the trained model.

    Args:
    - data_loader (DataLoader): DataLoader to fetch input data.
    - model (torch.nn.Module): The trained model.
    - device (torch.device): Device to run the model on.
    - modeltype (str): Model type ('VAE' or other).
    - num_images (int): Number of images to generate and plot.
    """
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        # Fetch a single batch of images
        for images, _ in data_loader:
            images = images.to(device)
            
            # Generate images using the model
            if modeltype == 'VAE':
                _, _, _, reconstructed = model(images)  # VAE forward pass
            else:
                raise ValueError(f"Unsupported model type: {modeltype}")

            # Plot the original and reconstructed images
            fig, axes = plt.subplots(2, num_images, figsize=(15, 4))
            for i in range(num_images):
                # Original image
                axes[0, i].imshow(images[i].permute(1, 2, 0).cpu().numpy(), cmap="gray")
                axes[0, i].axis("off")
                axes[0, i].set_title("Original")

                # Reconstructed image
                axes[1, i].imshow(reconstructed[i].permute(1, 2, 0).cpu().numpy(), cmap="gray")
                axes[1, i].axis("off")
                axes[1, i].set_title("Reconstructed")

            plt.tight_layout()
            plt.show()
            break  # Show only one batch
