

import torch
import matplotlib.pyplot as plt
import sys
import os
import requests

sys.path.append(os.getcwd())
from mamba_ssm.config import MambaConfig
from mamba_ssm.model import Mamba

def main():
    # Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = MambaConfig(d_model=128, n_layers=4, vocab_size=65, d_state=16, expand=2)
    model = Mamba(config).to(device)
    
    # Try to load weights
    if os.path.exists("mamba_shakespeare_final.pth"):
        model.load_state_dict(torch.load("mamba_shakespeare_final.pth", map_location=device))
        print("Loaded trained weights.")
    else:
        print("WARNING: No weights found. Plot will be random (just for demo structure).")

    # Load Vocab
    if not os.path.exists("data/shakespeare.txt"):
        print("Error: data/shakespeare.txt not found.")
        return
    with open("data/shakespeare.txt", "r") as f: text = f.read()
    chars = sorted(list(set(text)))
    stoi = { ch:i for i,ch in enumerate(chars) }
    
    # Input Sequence
    input_str = "All the world's a stage,\nAnd all the men and women merely players;"
    input_ids = torch.tensor([[stoi.get(c, 0) for c in input_str]]).to(device)
    
    # Forward Pass
    model.eval()
    with torch.no_grad():
        _ = model(input_ids)
        
    # Extract Delta from the first layer
    # Shape: [Batch, Length, D_inner]
    dt = model.layers[0].x_proj.last_dt.cpu()
    
    # Average over channel dimension to get a 1D "Focus Score" per token
    dt_magnitude = dt[0].mean(dim=-1).numpy()
    
    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(dt_magnitude, marker='o', color='orange')
    plt.xticks(range(len(input_str)), list(input_str), rotation=90, fontsize=8)
    plt.title("Mamba Selectivity: Magnitude of $\Delta$ (dt) per Character")
    plt.ylabel("Update Magnitude")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("delta_visualization.png")
    print("Saved delta_visualization.png")

if __name__ == "__main__":
    main()