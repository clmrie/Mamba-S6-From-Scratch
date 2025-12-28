import sys
import os
sys.stdout.reconfigure(line_buffering=True)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from mamba_ssm.utils import RMSNorm
from mamba_ssm.s6 import S6
from mamba_ssm.config import MambaConfig

class CausalMambaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.in_proj = nn.Linear(config.d_model, config.d_inner * 2, bias=config.bias)
        self.conv1d = nn.Conv1d(
            in_channels=config.d_inner, out_channels=config.d_inner,
            bias=config.conv_bias, kernel_size=config.d_conv,
            groups=config.d_inner, padding=config.d_conv - 1,
        )
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)
        self.forward_s6 = S6(config)

    def forward(self, x):
        x_and_res = self.in_proj(x)
        (x, res) = x_and_res.split(split_size=[self.config.d_inner, self.config.d_inner], dim=-1)
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)[:, :, :x.shape[-1]]
        x = x.permute(0, 2, 1)
        x = F.silu(x)
        
        x = self.forward_s6(x)
        
        res = F.silu(res)
        x = x * res
        return self.out_proj(x)

class CausalVisionMamba(nn.Module):
    def __init__(self, config, num_classes=10, channels=1):
        super().__init__()
        self.patch_embed = nn.Conv2d(channels, config.d_model, kernel_size=4, stride=4)
        self.layers = nn.ModuleList([CausalMambaBlock(config) for _ in range(config.n_layers)])
        self.norm_f = RMSNorm(config.d_model)
        self.head = nn.Linear(config.d_model, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        b, d, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        for layer in self.layers:
            x = x + layer(x)
        x = self.norm_f(x)
        x = x.mean(dim=1) 
        return self.head(x)

def main():
    print("Running CAUSAL (Unidirectional) Vision Mamba for Ablation Study...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    BATCH_SIZE = 64
    LR = 3e-4
    EPOCHS = 5
    RESULTS_DIR = "results_mnist"
    SCORE_FILE = os.path.join(RESULTS_DIR, "mnist_causal_scores.csv")
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    config = MambaConfig(d_model=64, n_layers=2, vocab_size=0, d_state=16, expand=2)
    model = CausalVisionMamba(config, num_classes=10, channels=1).to(device)
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    with open(SCORE_FILE, "w") as f:
        f.write("epoch,test_accuracy\n")
    
    for epoch in range(EPOCHS): 
        model.train()
        for i, (x, y) in enumerate(trainloader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
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
        
        acc = 100*correct/total
        print(f"Epoch {epoch+1} Causal Accuracy: {acc:.2f}%")
        
        with open(SCORE_FILE, "a") as f:
            f.write(f"{epoch+1},{acc:.2f}\n")

if __name__ == "__main__":
    main()