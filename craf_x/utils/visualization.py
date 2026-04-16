import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_tensor_as_image(ax, tensor: torch.Tensor, title: str):
    """Plots a 3-channel (C, H, W) tensor as an RGB image."""
    img = tensor.detach().cpu().numpy()
    if img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    # Normalize to [0,1] for display if needed
    vmin, vmax = img.min(), img.max()
    if vmax > vmin:
        img = (img - vmin) / (vmax - vmin)
    ax.imshow(img)
    ax.set_title(title)
    ax.axis('off')

def plot_heatmap(ax, tensor: torch.Tensor, title: str, cmap: str = 'magma'):
    """Plots a 1-channel (1, H, W) tensor as a spatial heatmap."""
    heatmap = tensor.detach().cpu().numpy().squeeze()
    im = ax.imshow(heatmap, cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.axis('off')
    return im
