"""
Diffusion Model Implementation (DDPM - Denoising Diffusion Probabilistic Model)
================================================================================
Demonstrates the forward and reverse diffusion processes:
  - Forward Process: Gradually adds Gaussian noise to an image over T timesteps
  - Reverse Process: A neural network learns to denoise step by step
  
Based on the paper "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class SinusoidalPositionEmbedding(nn.Module):
    """
    Encodes the diffusion timestep as a sinusoidal embedding.
    This helps the model understand which noise level it should denoise from.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class ResBlock(nn.Module):
    """Residual block with time embedding injection."""

    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.GroupNorm(8, out_ch)
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)

        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t_emb):
        h = F.silu(self.bn1(self.conv1(x)))
        # Add time embedding
        t = self.time_mlp(t_emb)[:, :, None, None]
        h = h + t
        h = F.silu(self.bn2(self.conv2(h)))
        return h + self.shortcut(x)


class SimpleUNet(nn.Module):
    """
    A simplified U-Net architecture for noise prediction.
    
    The U-Net takes a noisy image and a timestep, and predicts the noise
    that was added to the image at that timestep.
    """

    def __init__(self, in_channels=3, base_channels=64, time_emb_dim=128):
        super().__init__()

        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
        )

        # Encoder (Downsampling)
        self.down1 = ResBlock(in_channels, base_channels, time_emb_dim)
        self.pool1 = nn.Conv2d(base_channels, base_channels, 3, stride=2, padding=1)

        self.down2 = ResBlock(base_channels, base_channels * 2, time_emb_dim)
        self.pool2 = nn.Conv2d(base_channels * 2, base_channels * 2, 3, stride=2, padding=1)

        self.down3 = ResBlock(base_channels * 2, base_channels * 4, time_emb_dim)
        self.pool3 = nn.Conv2d(base_channels * 4, base_channels * 4, 3, stride=2, padding=1)

        # Bottleneck
        self.bottleneck = ResBlock(base_channels * 4, base_channels * 4, time_emb_dim)

        # Decoder (Upsampling)
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = ResBlock(base_channels * 8, base_channels * 2, time_emb_dim)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = ResBlock(base_channels * 4, base_channels, time_emb_dim)

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = ResBlock(base_channels * 2, base_channels, time_emb_dim)

        self.final = nn.Conv2d(base_channels, in_channels, 1)

    def forward(self, x, t):
        t_emb = self.time_embed(t)

        # Store original size for matching
        original_size = x.shape[2:]

        # Encoder
        d1 = self.down1(x, t_emb)
        d1p = self.pool1(d1)

        d2 = self.down2(d1p, t_emb)
        d2p = self.pool2(d2)

        d3 = self.down3(d2p, t_emb)
        d3p = self.pool3(d3)

        # Bottleneck
        bn = self.bottleneck(d3p, t_emb)

        # Decoder with skip connections
        u3 = self.up3(bn)
        u3 = F.interpolate(u3, size=d3.shape[2:], mode="bilinear", align_corners=False)
        u3 = self.dec3(torch.cat([u3, d3], dim=1), t_emb)

        u2 = self.up2(u3)
        u2 = F.interpolate(u2, size=d2.shape[2:], mode="bilinear", align_corners=False)
        u2 = self.dec2(torch.cat([u2, d2], dim=1), t_emb)

        u1 = self.up1(u2)
        u1 = F.interpolate(u1, size=d1.shape[2:], mode="bilinear", align_corners=False)
        u1 = self.dec1(torch.cat([u1, d1], dim=1), t_emb)

        out = self.final(u1)
        # Ensure output matches input size
        out = F.interpolate(out, size=original_size, mode="bilinear", align_corners=False)
        return out


class DiffusionModel:
    """
    DDPM (Denoising Diffusion Probabilistic Model)
    
    Manages the noise schedule and provides methods for:
      - Forward process (adding noise)
      - Reverse process (denoising)
      - Training
    """

    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, image_size=128, in_channels=3):
        self.num_timesteps = num_timesteps
        self.image_size = image_size
        self.in_channels = in_channels
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Ensure that the total amount of noise added remains consistent 
        # even if num_timesteps is changed from the default 1000.
        # If we don't scale it, at T=200 it won't reach pure noise!
        scale = 1000.0 / num_timesteps
        beta_end_scaled = beta_end * scale

        # Linear noise schedule
        self.betas = torch.linspace(beta_start, beta_end_scaled, num_timesteps).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alpha_cumprod_prev = F.pad(self.alpha_cumprod[:-1], (1, 0), value=1.0)

        # Precompute useful quantities
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)
        self.sqrt_recip_alpha = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1.0 - self.alpha_cumprod_prev) / (1.0 - self.alpha_cumprod)

        # Model
        self.model = SimpleUNet(
            in_channels=in_channels,
            base_channels=64,
            time_emb_dim=128
        ).to(self.device)

    def forward_process(self, x_0, t, noise=None):
        """
        Forward diffusion: q(x_t | x_0)
        
        Adds noise to x_0 according to the noise schedule at timestep t.
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha_cumprod_t = self.sqrt_alpha_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod[t][:, None, None, None]

        x_t = sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise
        return x_t, noise

    def get_forward_process_visualization(self, x_0, num_steps=10):
        """
        Generate a series of images showing the forward diffusion process.
        
        Args:
            x_0: original image tensor (1, C, H, W)
            num_steps: number of steps to visualize
        
        Returns:
            list of (timestep, image_numpy) tuples
        """
        x_0 = x_0.to(self.device)
        x_0 = x_0 * 2.0 - 1.0  # Scale [0, 1] to [-1, 1]
        results = [(0, ((x_0 + 1.0) / 2.0).squeeze(0).cpu().permute(1, 2, 0).numpy())]

        timesteps = np.linspace(0, self.num_timesteps - 1, num_steps, dtype=int)

        with torch.no_grad():
            for t_val in timesteps[1:]:
                t = torch.tensor([t_val]).long().to(self.device)
                x_t, _ = self.forward_process(x_0, t)
                img = x_t.squeeze(0).cpu().permute(1, 2, 0).numpy()
                img = (img + 1.0) / 2.0  # Scale [-1, 1] back to [0, 1]
                img = np.clip(img, 0, 1)
                results.append((t_val, img))

        return results

    def train_step(self, x_0):
        """Single training step: predict the noise added to x_0."""
        self.model.train()
        x_0 = x_0.to(self.device)
        
        # Scale to [-1, 1] for diffusion
        x_0 = x_0 * 2.0 - 1.0
        
        # We simulate a "batch" of identical images to allow the model to see different 
        # timesteps in a single optimization step, which is crucial for convergence.
        batch_size = 16
        x_0 = x_0.repeat(batch_size, 1, 1, 1)

        # Sample random timestep
        t = torch.randint(0, self.num_timesteps, (x_0.shape[0],)).long().to(self.device)

        # Add noise
        noise = torch.randn_like(x_0)
        x_t, _ = self.forward_process(x_0, t, noise)

        # Predict noise
        predicted_noise = self.model(x_t, t)

        # Loss
        loss = F.mse_loss(predicted_noise, noise)
        return loss

    @torch.no_grad()
    def reverse_step(self, x_t, t):
        """
        Single reverse diffusion step: p(x_{t-1} | x_t)
        
        Uses the trained model to predict noise and remove it.
        """
        self.model.eval()

        t_tensor = torch.tensor([t]).long().to(self.device)

        # Predict noise
        predicted_noise = self.model(x_t, t_tensor)

        # Compute x_{t-1}
        alpha_t = self.alphas[t]
        alpha_cumprod_t = self.alpha_cumprod[t]
        beta_t = self.betas[t]

        # Mean of the reverse process
        coeff = beta_t / self.sqrt_one_minus_alpha_cumprod[t]
        mean = self.sqrt_recip_alpha[t] * (x_t - coeff * predicted_noise)

        if t > 0:
            noise = torch.randn_like(x_t)
            sigma = torch.sqrt(self.posterior_variance[t])
            x_prev = mean + sigma * noise
        else:
            x_prev = mean

        return x_prev

    @torch.no_grad()
    def sample(self, num_samples=1, progress_callback=None):
        """
        Generate images by running the full reverse diffusion process.
        Start from pure noise and iteratively denoise.
        """
        self.model.eval()

        x = torch.randn(num_samples, self.in_channels, self.image_size, self.image_size).to(self.device)
        intermediates = []

        for t in reversed(range(self.num_timesteps)):
            x = self.reverse_step(x, t)

            # Save intermediate steps for visualization
            if t % (self.num_timesteps // 10) == 0 or t == 0:
                img = x.squeeze(0).cpu().permute(1, 2, 0).numpy()
                img = (img + 1.0) / 2.0  # Scale back to [0, 1] from [-1, 1]
                img = np.clip(img, 0, 1)
                intermediates.append((t, img))

            if progress_callback:
                progress_callback(self.num_timesteps - t, self.num_timesteps)
        
        # Final image to [0, 1]
        x = (x + 1.0) / 2.0
        return x, intermediates

    def train_on_image(self, image_tensor, num_epochs=500, lr=0.001, progress_callback=None):
        """
        Train the diffusion model on a single image (for demonstration).
        
        This overfits on purpose to show the denoising capability.
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        image_tensor = image_tensor.to(self.device)

        history = []

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            loss = self.train_step(image_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            loss_val = loss.item()
            history.append({"epoch": epoch + 1, "loss": loss_val})

            if progress_callback:
                progress_callback(epoch + 1, num_epochs, {"loss": loss_val})

        return history


def get_noise_schedule_visualization(num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
    """
    Generate data for visualizing the noise schedule.
    
    Returns:
        dict with arrays for betas, alphas, alpha_cumprod, etc.
    """
    betas = np.linspace(beta_start, beta_end, num_timesteps)
    alphas = 1.0 - betas
    alpha_cumprod = np.cumprod(alphas)

    return {
        "timesteps": np.arange(num_timesteps),
        "betas": betas,
        "alphas": alphas,
        "alpha_cumprod": alpha_cumprod,
        "sqrt_alpha_cumprod": np.sqrt(alpha_cumprod),
        "sqrt_one_minus_alpha_cumprod": np.sqrt(1 - alpha_cumprod),
    }
