# train_shakespeare.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import requests
import os
import sys

# Ensure we can find the mamba_ssm package
sys.path.append(os.getcwd())

from mamba_ssm.config import MambaConfig
from mamba_ssm.model import Mamba

class CharDataset(Dataset):
    def __init__(self, data, block_size):
        chars = sorted(list(set(data)))
        self.vocab_size = len(chars)
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.data = [self.stoi[c] for c in data]
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def get_data():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    fn = "shakespeare.txt"
    if not os.path.exists(fn):
        with open(fn, "w") as f:
            f.write(requests.get(url).text)
    with open(fn, "r") as f:
        return f.read()

# --- BETTER GENERATION FUNCTION ---
def generate_text(model, idx, max_new_tokens=100, temperature=1.0, top_k=None):
    # This is a robust generation loop for validation
    for _ in range(max_new_tokens):
        # Crop context if it gets too long for the block size
        idx_cond = idx if idx.size(1) <= 128 else idx[:, -128:]
        
        # Forward pass
        logits = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        
        # Optional: Top-K filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        # Sample from distribution (NOT ARGMAX)
        probs = nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # Append
        idx = torch.cat((idx, idx_next), dim=1)
        
    return idx
# ----------------------------------

def main():
    text = get_data()
    block_size = 128
    dataset = CharDataset(text, block_size)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    config = MambaConfig(
        d_model=128,
        n_layers=4,
        vocab_size=dataset.vocab_size,
        d_state=16,
        expand=2
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    model = Mamba(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)

    model.train()
    # Train for longer to get good results
    for epoch in range(10): 
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            logits = model(x)
            loss = nn.functional.cross_entropy(logits.view(-1, dataset.vocab_size), y.view(-1))
            
            loss.backward()
            
            # Gradient Clipping is essential for Mamba stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            if i % 100 == 0:
                print(f"Epoch {epoch} | Step {i} | Loss {loss.item():.4f}")
            
            # Generate sample every 500 steps
            if i > 0 and i % 500 == 0:
                model.eval()
                with torch.no_grad():
                    context = torch.tensor([[dataset.stoi['\n']]], device=device)
                    # Use Temperature 0.8 for "safe" but creative text
                    out = generate_text(model, context, max_new_tokens=200, temperature=0.8)
                    decoded = "".join([dataset.itos[i.item()] for i in out[0]])
                    print("\n=========================================")
                    print(decoded)
                    print("=========================================\n")
                model.train()

if __name__ == "__main__":
    main()