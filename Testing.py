import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.image_paths = [os.path.join(directory, fname) for fname in os.listdir(directory) if fname.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")  # Open image and ensure it's in RGB mode
        if self.transform:
            img = self.transform(img)
        return img, os.path.basename(img_path)  # Return the image and its filename

# VAE Model
class VAE(nn.Module):
    def __init__(self, img_size=640, latent_dim=256):
        super(VAE, self).__init__()

        self.img_size = img_size
        self.latent_dim = latent_dim
        
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.ReLU()
        )

        # Calculate the flattened size dynamically based on the encoder output
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, img_size, img_size)
            encoded_output = self.encoder(dummy_input)
            self.flattened_size = encoded_output.view(1, -1).shape[1]  # Auto-detect correct size

        # Latent space
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)
        
        # Decoder network
        self.fc_decode = nn.Linear(latent_dim, self.flattened_size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.Sigmoid()  # To ensure the output is in the range [0, 1]
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)  # Flatten the encoded features
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def decode(self, z):
        x = self.fc_decode(z)
        x = x.view(-1, 512, 40, 40)  # Adjust the reshape size based on the encoder's output
        x = self.decoder(x)
        return x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

# Loss Function (VAE)
def loss_function(recon_x, x, mu, logvar):
    MSE = nn.functional.mse_loss(recon_x.view(-1, 3 * 640 * 640), x.view(-1, 3 * 640 * 640), reduction='sum')
    # KL divergence loss
    return MSE + 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1 - logvar)

# Data transform
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

# Inference function with filename support
def infer(model, dataloader):
    model.eval()
    with torch.no_grad():
        for idx, (img, filename) in enumerate(dataloader):
            img = img.to(device)
            recon, mu, logvar = model(img)
            
            # Calculate the reconstruction error (Mean Squared Error)
            recon_error = nn.functional.mse_loss(recon.view(-1, 3 * 640 * 640), img.view(-1, 3 * 640 * 640), reduction='none')
            recon_error = recon_error.view(img.shape[0], 3, 640, 640)  # Reshape to the original image shape

            # Calculate the pixel-wise reconstruction error
            recon_error = recon_error.mean(dim=1)  # Average error across color channels (RGB)

            # Find regions with higher reconstruction error
            threshold = recon_error.mean() + 2 * recon_error.std()  # Set a threshold for anomaly
            anomaly_mask = recon_error > threshold  # Create a mask for anomalies

            # Visualize the result and save with the filename
            visualize_anomalies(img[0], recon[0], anomaly_mask[0], filename[0])  # Pass filename[0]

# Visualization function to show anomalies
def visualize_anomalies(original_img, reconstructed_img, anomaly_mask, filename):
    # Convert tensors to numpy arrays for visualization
    original_img = original_img.cpu().numpy().transpose(1, 2, 0)
    reconstructed_img = reconstructed_img.cpu().numpy().transpose(1, 2, 0)
    anomaly_mask = anomaly_mask.cpu().numpy()

    # Visualize anomalies as green regions over the original image
    anomaly_overlay = np.copy(original_img)
    anomaly_overlay[anomaly_mask] = [0, 1, 0]  # Mark anomalies in green
    
    # Blend the anomaly regions with the original image
    alpha = 0.6  # Adjust transparency
    anomaly_overlay = (1 - alpha) * original_img + alpha * anomaly_overlay

    # Save the masked image with the original filename
    masked_images_path = "masked_v2"
    masked_image_path = os.path.join(masked_images_path, filename)  # Use the original filename
    # Check if the directory exists, if not, create it
    if not os.path.exists(masked_images_path):
        os.makedirs(masked_images_path)
    plt.imsave(masked_image_path, anomaly_overlay)

    # Plot the original image, reconstructed image, and anomaly mask (optional visualization)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    ax[0].imshow(original_img)
    ax[0].set_title("Original Image")
    ax[0].axis('off')

    ax[1].imshow(reconstructed_img)
    ax[1].set_title("Reconstructed Image")
    ax[1].axis('off')

    ax[2].imshow(anomaly_overlay)
    ax[2].set_title("Anomalous Regions")
    ax[2].axis('off')

    plt.show()

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model
vae = VAE(img_size=640, latent_dim=256).to(device)

# Load the pre-trained model
vae.load_state_dict(torch.load('vae_model.pth', weights_only=True))
vae.to(device)

# Dataset and DataLoader for testing
test_dataset = ImageDataset("test_v2_cropped", transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Run inference on test set
infer(vae, test_dataloader)