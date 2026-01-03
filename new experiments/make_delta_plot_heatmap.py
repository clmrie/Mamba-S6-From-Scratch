import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
import os
import torchvision
import torchvision.transforms as transforms
import cv2
import numpy as np

# Ensure the import path is correct for your project structure
sys.path.append(os.getcwd())
from mamba_ssm.config import MambaConfig
from vision.model import VisionMamba

def main():
    # 1. Initialize Model and Environment
    print("🚀 Loading Vision Mamba model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Configuration must match your training parameters
    config = MambaConfig(d_model=64, n_layers=2, vocab_size=0, d_state=16, expand=2)
    model = VisionMamba(config, num_classes=10, channels=1).to(device)
    
    ckpt_path = "checkpoints/mnist/vision_mamba_mnist_epoch_5.pth"
    if not os.path.exists(ckpt_path):
        print(f"❌ Error: Checkpoint not found at {ckpt_path}")
        return
        
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    # 2. Load MNIST Dataset
    print("📂 Loading MNIST test dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # 3. Collect one sample for each digit from 0 to 9
    samples = {}
    for img, label in dataset:
        if label not in samples:
            samples[label] = img
        if len(samples) == 10:
            break
    
    # 4. Setup Plotting: 2 rows and 5 columns
    fig, axes = plt.subplots(2, 5, figsize=(20, 9))
    fig.suptitle("Vision Mamba Interpretability: Delta ($\Delta$) Heatmaps for Digits 0-9", fontsize=22, fontweight='bold')

    print("🎨 Generating heatmaps for all digits...")
    for digit in range(10):
        image_tensor = samples[digit].unsqueeze(0).to(device)
        ax = axes[digit // 5, digit % 5]
        
        # Forward pass to populate 'last_dt' inside the s6 modules
        with torch.no_grad():
            _ = model(image_tensor)
        
        try:
            # Extract Delta values from both forward and backward scans
            dt_fwd = model.layers[0].forward_s6.last_dt.cpu()
            dt_bwd = model.layers[0].backward_s6.last_dt.cpu()
            
            # Calculate intensity scores by averaging over the inner dimension
            score_fwd = dt_fwd.mean(dim=-1).squeeze()
            # Flip backward scan results to align them spatially with the forward scan
            score_bwd_aligned = torch.flip(dt_bwd.mean(dim=-1).squeeze(), dims=[0])
            
            # Fusion Strategy: Use the Maximum value between scans
            # This highlights edges more effectively than simple averaging
            dt_score = torch.maximum(score_fwd, score_bwd_aligned)
            
            # Normalize to [0, 1] range for visualization
            dt_norm = (dt_score - dt_score.min()) / (dt_score.max() - dt_score.min() + 1e-8)
            
            # Reshape 1D sequence (49) back to 2D spatial structure (7x7)
            heatmap_small = dt_norm.reshape(7, 7).numpy()
            
            # Upscale 7x7 to 28x28 using cubic interpolation for a smooth overlay
            heatmap_large = cv2.resize(heatmap_small, (28, 28), interpolation=cv2.INTER_CUBIC)
            
            # Plot the original grayscale digit as the background
            ax.imshow(samples[digit].squeeze(), cmap='gray')
            
            # Overlay the fused heatmap using the 'jet' colormap
            im = ax.imshow(heatmap_large, cmap='jet', alpha=0.5)
            
            ax.set_title(f"Target Digit: {digit}", fontsize=14, fontweight='bold')
            ax.axis('off')
            
        except AttributeError:
            ax.text(0.5, 0.5, "AttributeError: last_dt not found", ha='center', color='red')
            ax.axis('off')

    # Add a colorbar to indicate update intensity levels
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='State Update Intensity ($\Delta$)')

    # 5. Save the final figure
    output_name = 'mamba_all_digits_interpretability.png'
    plt.savefig(output_name, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ Success! Results saved to: {output_name}")

if __name__ == "__main__":
    main()