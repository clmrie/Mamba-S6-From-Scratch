

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Force immediate printing
sys.stdout.reconfigure(line_buffering=True)

def main():
    # 1. Check if we have the Mamba logs to compare against
    mamba_log_file = "results_mnist/mnist_training_log.csv"
    if not os.path.exists(mamba_log_file):
        print(f"Error: Could not find {mamba_log_file}. Did you run train_mnist.py?")
        return

    print("--- STEP 1: Training Vanilla RNN Baseline (Fast) ---")
    BATCH_SIZE = 64
    EPOCHS = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Standard RNN
    class VanillaRNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.RNN(input_size=16, hidden_size=64, num_layers=2, batch_first=True)
            self.fc = nn.Linear(64, 10)
        def forward(self, x):
            x = x.unfold(2, 4, 4).unfold(3, 4, 4).reshape(x.size(0), -1, 16)
            out, _ = self.rnn(x)
            return self.fc(out[:, -1, :])

    model = VanillaRNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    rnn_history = []
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for i, (x, y) in enumerate(trainloader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 10 == 9:
                # Store step and loss
                global_step = epoch * len(trainloader) + i
                rnn_history.append({'global_step': global_step, 'loss': running_loss / 10})
                running_loss = 0.0
        print(f"RNN Epoch {epoch+1}/{EPOCHS} complete.")

    # 2. Plotting
    print("--- STEP 2: Generating Comparison Plot ---")
    df_rnn = pd.DataFrame(rnn_history)
    df_mamba = pd.read_csv(mamba_log_file)
    
    # Normalize Mamba steps
    batches_per_epoch = 938
    df_mamba['global_step'] = (df_mamba['epoch'] - 1) * batches_per_epoch + df_mamba['batch']
    
    plt.figure(figsize=(10, 6))
    plt.plot(df_mamba['global_step'], df_mamba['loss'], label='Vision Mamba (Ours)', color='#1f77b4', alpha=0.9)
    plt.plot(df_rnn['global_step'], df_rnn['loss'], label='Vanilla RNN (Baseline)', color='#d62728', alpha=0.7, linestyle='--')
    
    plt.title('Convergence: Mamba vs Vanilla RNN on MNIST')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 2.5)
    
    plt.savefig('mamba_vs_rnn.png', dpi=300)
    print("✅ Saved mamba_vs_rnn.png")

if __name__ == "__main__":
    main()
    