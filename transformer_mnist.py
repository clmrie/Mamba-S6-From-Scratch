import sys
import os
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# ==========================================
# 1. Hand-Crafted Transformer Model
# ==========================================
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, patch_size=4, emb_size=64, img_size=28):
        super().__init__()
        self.patch_size = patch_size
        # 使用 Conv2d 实现非重叠 Patch 切分和投影
        self.projection = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) ** 2

    def forward(self, x):
        # x: (B, C, H, W) -> (B, E, H/P, W/P) -> (B, E, N) -> (B, N, E)
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        # qkv: (B, N, 3*C) -> (B, N, 3, Heads, C//Heads) -> (3, B, Heads, N, C//Heads)
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention calculation
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Weighted sum
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads=heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_dim, dropout=dropout)

    def forward(self, x):
        # Pre-Norm 结构 (更稳定)
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class SimpleViT(nn.Module):
    def __init__(self, img_size=28, patch_size=4, in_channels=1, num_classes=10, 
                 d_model=64, depth=2, heads=4, mlp_dim=128, dropout=0.1):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(in_channels, patch_size, d_model, img_size)
        num_patches = self.patch_embed.num_patches
        
        # Class token & Position Embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, d_model))
        self.pos_drop = nn.Dropout(dropout)

        # Transformer Encoder Layers
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, heads, mlp_dim, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add Position Embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        
        # Use CLS token for classification
        cls_token_final = x[:, 0]
        return self.head(cls_token_final)

# ==========================================
# 2. Main Training Script
# ==========================================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    # --- Hyperparameters (Aligned with Mamba) ---
    BATCH_SIZE = 64
    LR = 1e-3
    EPOCHS = 5
    
    # Save dirs (separate from Mamba to avoid overwrite)
    SAVE_DIR = "checkpoints_mnist_transformer"
    RESULTS_DIR = "results_mnist_transformer"
    LOG_FILE = os.path.join(RESULTS_DIR, "mnist_training_log.csv")
    SCORE_FILE = os.path.join(RESULTS_DIR, "mnist_scores.csv")

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Transformer on: {device}")

    # --- Data Loading (Same as Mamba) ---
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    print("Loading MNIST Data...")
    # reuse ./data to save download time
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # --- Model Initialization ---
    # Config aligned with Mamba: d_model=64, layers=2
    # Patch size 4 -> sequence length 49 (7x7)
    model = SimpleViT(
        img_size=28, 
        patch_size=4, 
        in_channels=1, 
        num_classes=10,
        d_model=64,      # Same as Mamba
        depth=2,         # Same as Mamba
        heads=4,         # 64 / 4 = 16 dim per head
        mlp_dim=128,     # FeedForward dim (usually 2x-4x model dim)
        dropout=0.1
    ).to(device)

    params = count_parameters(model)
    print(f"Transformer Model Initialized. Total Trainable Params: {params:,}")
    # (Typical Mamba roughly 20-40k for these specs, this Transformer should be similar)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # --- Logging Init ---
    with open(LOG_FILE, "w") as f:
        f.write("epoch,batch,loss,time_per_10_steps\n")
        
    with open(SCORE_FILE, "w") as f:
        f.write("epoch,avg_epoch_loss,test_accuracy\n")
    
    print("Starting Training Loop (Transformer)...")
    
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
        
        checkpoint_path = os.path.join(SAVE_DIR, f"transformer_mnist_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)

    print("Transformer Training Complete. Results saved in results_mnist_transformer/")

if __name__ == "__main__":
    main()