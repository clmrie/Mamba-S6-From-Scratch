
import sys
import os
sys.stdout.reconfigure(line_buffering=True)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import time
import csv

sys.path.append(os.getcwd())

from mamba_ssm.config import MambaConfig
from vision.model import VisionMamba


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    BATCH_SIZE = 64
    LR = 1e-3
    EPOCHS = 5
    SAVE_DIR = "checkpoints_mnist"
    RESULTS_DIR = "results_mnist"
    LOG_FILE = os.path.join(RESULTS_DIR, "mnist_training_log.csv")
    SCORE_FILE = os.path.join(RESULTS_DIR, "mnist_scores.csv")

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    config = MambaConfig(
        d_model=64, 
        n_layers=2, 
        vocab_size=0, 
        d_state=16, 
        expand=2
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Vision Mamba on: {device}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    print("Loading MNIST Data...")
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = VisionMamba(config, num_classes=10, channels=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    with open(LOG_FILE, "w") as f:
        f.write("epoch,batch,loss,time_per_10_steps\n")
        
    with open(SCORE_FILE, "w") as f:
        f.write("epoch,avg_epoch_loss,test_accuracy\n")
    
    params = count_parameters(model)
    print(f"Model Initialized. Total Trainable Params: {params:,}")

    print("Starting Training Loop...")
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_running_loss = 0.0
        step_running_loss = 0.0
        start_time = time.time()
        
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_running_loss += loss.item()
            step_running_loss += loss.item()
            
            if i % 10 == 9:
                elapsed = time.time() - start_time
                avg_step_loss = step_running_loss / 10
                
                print(f"[Epoch {epoch+1}, Batch {i+1}] Loss: {avg_step_loss:.4f} | Time: {elapsed:.2f}s")
                
                with open(LOG_FILE, "a") as f:
                    f.write(f"{epoch+1},{i+1},{avg_step_loss:.4f},{elapsed:.4f}\n")
                
                step_running_loss = 0.0
                start_time = time.time()

        print(f"Validating Epoch {epoch+1}...")
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        avg_epoch_loss = epoch_running_loss / len(trainloader)
        
        print(f"==> Epoch {epoch+1} | Loss: {avg_epoch_loss:.4f} | Acc: {acc:.2f}%")
        
        with open(SCORE_FILE, "a") as f:
            f.write(f"{epoch+1},{avg_epoch_loss:.4f},{acc:.2f}\n")
        
        checkpoint_path = os.path.join(SAVE_DIR, f"vision_mamba_mnist_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)

    print("Extension Task Complete. Results saved in results_mnist/")

if __name__ == "__main__":
    main()
    