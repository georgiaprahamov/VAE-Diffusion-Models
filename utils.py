"""
Utility functions for image processing and visualization.
"""

import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import io


def preprocess_image(uploaded_file, target_size=128):
    """
    Load and preprocess an uploaded image for model input.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        target_size: resize target (square)
    
    Returns:
        image_tensor: torch.Tensor (1, 3, H, W) in [0, 1]
        original_image: PIL Image (original)
        resized_image: PIL Image (resized)
    """
    original_image = Image.open(uploaded_file).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
    ])

    resized_image = original_image.resize((target_size, target_size), Image.LANCZOS)
    image_tensor = transform(original_image).unsqueeze(0)

    return image_tensor, original_image, resized_image


def tensor_to_pil(tensor):
    """Convert a tensor (1, C, H, W) or (C, H, W) to PIL Image."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    img = tensor.cpu().permute(1, 2, 0).numpy()
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(img)


def numpy_to_pil(arr):
    """Convert a numpy array (H, W, C) in [0, 1] to PIL Image."""
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)
