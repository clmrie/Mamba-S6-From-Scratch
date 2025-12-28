import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import os
import sys

# Force immediate printing
sys.stdout.reconfigure(line_buffering=True)

def main():
    BATCH_SIZE = 64
    EPOCHS = 5
    LR = 1e-3
    
    # Ensure results folder exists
    RESULTS_DIR = "results_mnist"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    SCORE_FILE = os.path.join(RESULTS_DIR, "rnn_scores.csv")
    LOG_FILE = os.path.join(RESULTS_DIR, "rnn_training_log.csv")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Vanilla RNN Baseline on {device}...")
    
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Standard RNN Architecture (Matching input/output dims of Mamba)
    class VanillaRNN(nn.Module):
        def __init__(self):
            super().__init__()
            # Sequence length 49 (7x7 patches of 4x4 pixels)
            self.input_dim = 16 
            self.hidden_dim = 64 
            self.rnn = nn.RNN(self.input_dim, self.hidden_dim, num_layers=2, batch_first=True)
            self.fc = nn.Linear(self.hidden_dim, 10)

        def forward(self, x):
            # Flatten image into sequence of patches
            B, C, H, W = x.shape
            x = x.unfold(2, 4, 4).unfold(3, 4, 4)
            x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
            x = x.view(B, -1, 16)
            
            out, _ = self.rnn(x)
            return self.fc(out[:, -1, :])

    model = VanillaRNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # Init Log Files
    with open(LOG_FILE, "w") as f:
        f.write("epoch,batch,loss\n")
    with open(SCORE_FILE, "w") as f:
        f.write("epoch,test_accuracy\n")

    print("Starting RNN Training...")
    
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
                avg_loss = running_loss / 10
                # print(f"[Epoch {epoch+1}, Batch {i+1}] RNN Loss: {avg_loss:.4f}")
                with open(LOG_FILE, "a") as f:
                    f.write(f"{epoch+1},{i+1},{avg_loss:.4f}\n")
                running_loss = 0.0
        
        # --- CALCULATE REAL ACCURACY ---
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in testloader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                _, pred = torch.max(out.data, 1)
                total += y.size(0)
                correct += (pred == y).sum().item()
        
        acc = 100 * correct / total
        print(f"Epoch {epoch+1} RNN Test Accuracy: {acc:.2f}%")
        
        # Save to file
        with open(SCORE_FILE, "a") as f:
            f.write(f"{epoch+1},{acc:.2f}\n")

if __name__ == "__main__":
    main()