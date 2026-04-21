"""
Variational Autoencoder (VAE) Implementation
=============================================
A convolutional VAE that can encode images into a latent space
and decode them back, demonstrating the core concepts of VAEs:
  - Encoder: maps input -> latent distribution (mu, log_var)
  - Reparameterization trick: samples from the distribution
  - Decoder: maps latent vector -> reconstructed image
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Encoder(nn.Module):
    """Encodes input images into latent space parameters (mu, log_var)."""

    def __init__(self, in_channels=3, latent_dim=128):
        super().__init__()
        # Convolutional layers to extract features
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # Adaptive pooling to handle any input size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Fully connected layers for mu and log_var
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_log_var = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)

        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var


class Decoder(nn.Module):
    """Decodes latent vectors back into images."""

    def __init__(self, latent_dim=128, out_channels=3, output_size=128):
        super().__init__()
        self.output_size = output_size

        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)

        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.deconv4 = nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 256, 4, 4)
        x = F.leaky_relu(self.bn1(self.deconv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.deconv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.deconv3(x)), 0.2)
        x = torch.sigmoid(self.deconv4(x))

        # Resize to target output size
        x = F.interpolate(x, size=(self.output_size, self.output_size), mode="bilinear", align_corners=False)
        return x


class VAE(nn.Module):
    """
    Full Variational Autoencoder.
    
    Architecture:
      Input Image -> Encoder -> (mu, log_var) -> Reparameterize -> z -> Decoder -> Reconstructed Image
    
    Loss = Reconstruction Loss (BCE) + KL Divergence
    """

    def __init__(self, in_channels=3, latent_dim=128, image_size=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size

        self.encoder = Encoder(in_channels, latent_dim)
        self.decoder = Decoder(latent_dim, in_channels, image_size)

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick:
        z = mu + sigma * epsilon, where epsilon ~ N(0, 1)
        
        This allows gradients to flow through the sampling step.
        """
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        z = mu + std * epsilon
        return z

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decoder(z)
        return reconstruction, mu, log_var, z

    def encode(self, x):
        """Encode an image and return the latent vector."""
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var

    def decode(self, z):
        """Decode a latent vector into an image."""
        return self.decoder(z)

    @staticmethod
    def loss_function(reconstruction, original, mu, log_var, kl_weight=1.0):
        """
        VAE Loss = Reconstruction Loss + KL Divergence
        
        Reconstruction Loss: measures how well the decoder reconstructs the input
        KL Divergence: regularizes the latent space to be close to N(0, 1)
        """
        # Reconstruction loss (pixel-wise)
        recon_loss = F.mse_loss(reconstruction, original, reduction="sum")

        # KL Divergence: D_KL(q(z|x) || p(z))
        # = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        return recon_loss + kl_weight * kl_loss, recon_loss, kl_loss


def train_vae_on_image(image_tensor, latent_dim=128, num_epochs=300, lr=0.001, 
                       kl_weight=0.5, progress_callback=None):
    """
    Train a VAE on a single image (overfitting intentionally for demonstration).
    
    Args:
        image_tensor: torch.Tensor of shape (1, C, H, W)
        latent_dim: dimension of the latent space
        num_epochs: number of training epochs
        lr: learning rate
        kl_weight: weight for KL divergence loss
        progress_callback: optional callable(epoch, total, loss_dict) for progress
    
    Returns:
        model: trained VAE model
        history: list of loss dictionaries per epoch
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    channels = image_tensor.shape[1]
    image_size = image_tensor.shape[2]

    model = VAE(in_channels=channels, latent_dim=latent_dim, image_size=image_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    image_tensor = image_tensor.to(device)
    history = []

    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        reconstruction, mu, log_var, z = model(image_tensor)

        total_loss, recon_loss, kl_loss = VAE.loss_function(
            reconstruction, image_tensor, mu, log_var, kl_weight
        )

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        loss_dict = {
            "epoch": epoch + 1,
            "total_loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item(),
        }
        history.append(loss_dict)

        if progress_callback:
            progress_callback(epoch + 1, num_epochs, loss_dict)

    model.eval()
    return model, history


def get_latent_interpolation(model, z1, z2, steps=10):
    """
    Interpolate between two latent vectors and decode each step.
    
    Args:
        model: trained VAE
        z1, z2: latent vectors
        steps: number of interpolation steps
    
    Returns:
        list of decoded images as numpy arrays
    """
    device = next(model.parameters()).device
    images = []

    with torch.no_grad():
        for alpha in np.linspace(0, 1, steps):
            z_interp = (1 - alpha) * z1 + alpha * z2
            decoded = model.decode(z_interp.to(device))
            img = decoded.squeeze(0).cpu().permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            images.append(img)

    return images


def get_latent_space_samples(model, num_samples=16, latent_dim=128):
    """
    Generate images by sampling random points from the latent space.
    
    Args:
        model: trained VAE
        num_samples: number of samples to generate
        latent_dim: dimension of latent space
    
    Returns:
        list of generated images as numpy arrays
    """
    device = next(model.parameters()).device
    images = []

    with torch.no_grad():
        for _ in range(num_samples):
            z = torch.randn(1, latent_dim).to(device)
            decoded = model.decode(z)
            img = decoded.squeeze(0).cpu().permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            images.append(img)

    return images
