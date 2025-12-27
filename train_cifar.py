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

def main():
    BATCH_SIZE = 64
    LR = 1e-3
    EPOCHS = 5
    LOG_FILE = "vision_training_log.csv"

    config = MambaConfig(
        d_model=128, 
        n_layers=4, 
        vocab_size=0, 
        d_state=16, 
        expand=2
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Vision Mamba on: {device}")
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    print("Loading CIFAR-10 Data (This might take a moment if downloading)...")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = VisionMamba(config, num_classes=10).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    with open(LOG_FILE, "w") as f:
        f.write("epoch,batch,loss,time_per_10_steps\n")
    
    print("Starting Training Loop...")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 10 == 9:
                elapsed = time.time() - start_time
                avg_loss = running_loss / 10
                
                print(f"[Epoch {epoch+1}, Batch {i+1}] Loss: {avg_loss:.4f} | Time (10 batches): {elapsed:.2f}s")
                
                with open(LOG_FILE, "a") as f:
                    f.write(f"{epoch+1},{i+1},{avg_loss:.4f},{elapsed:.4f}\n")
                
                running_loss = 0.0
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
        print(f"==> Epoch {epoch+1} Test Accuracy: {acc:.2f}%")
        
        torch.save(model.state_dict(), f"vision_mamba_epoch_{epoch+1}.pth")

    print("Extension Task Complete.")

if __name__ == "__main__":
    main()