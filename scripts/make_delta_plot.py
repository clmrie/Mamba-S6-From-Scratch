
import torch
import matplotlib.pyplot as plt
import sys
import os
import torchvision
import torchvision.transforms as transforms

sys.path.append(os.getcwd())
from mamba_ssm.config import MambaConfig
from vision.model import VisionMamba

def main():
    # 1. Load the Trained Model
    print("Loading model checkpoint...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = MambaConfig(d_model=64, n_layers=2, vocab_size=0, d_state=16, expand=2)
    model = VisionMamba(config, num_classes=10, channels=1).to(device)
    
    # Path to your saved model (Epoch 5)
    ckpt_path = "checkpoints/mnist/vision_mamba_mnist_epoch_5.pth"
    if not os.path.exists(ckpt_path):
        print(f"Error: {ckpt_path} not found. Check your checkpoints folder.")
        return
        
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    # 2. Get a single image from MNIST
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    image, label = dataset[0] # Take the first image (usually a '7')
    image = image.unsqueeze(0).to(device)
    
    # 3. Forward pass to populate 'last_dt'
    with torch.no_grad():
        _ = model(image)
    
    # 4. Extract Delta from the first layer's Forward Scan
    # The block structure is: model.layers[0].forward_s6.last_dt
    try:
        dt = model.layers[0].forward_s6.last_dt.cpu()
        # Shape is [Batch, Length, D_inner] -> [1, 49, 128]
        # Average over D_inner to get a single "importance score" per patch
        dt_score = dt.mean(dim=-1).squeeze()
    except AttributeError:
        print("Error: Could not find 'last_dt'. Did you update s6.py with 'self.last_dt = dt'?")
        return

    # 5. Plotting
    print("Generating visualization...")
    plt.figure(figsize=(10, 5))
    
    # Subplot 1: The Digit
    plt.subplot(1, 2, 1)
    plt.imshow(image.cpu().squeeze(), cmap='gray')
    plt.title(f"Input Digit: {label}")
    plt.axis('off')
    
    # Subplot 2: The Mamba Selection (Delta)
    plt.subplot(1, 2, 2)
    plt.plot(dt_score.numpy(), marker='o', color='purple')
    plt.title("Mamba 'Focus' (Delta Magnitude)")
    plt.xlabel("Image Patch Sequence (0-49)")
    plt.ylabel("State Update Size ($\Delta$)")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('delta_visualization.png', dpi=300)
    print("✅ Saved delta_visualization.png")

if __name__ == "__main__":
    main()
    