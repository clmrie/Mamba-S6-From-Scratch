
# tasks/copy_task.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mamba_ssm.config import MambaConfig
from mamba_ssm.model import Mamba

class SelectiveCopyDataset(Dataset):
    def __init__(self, vocab_size, seq_len, num_samples):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        data = torch.randint(0, self.vocab_size - 2, (self.seq_len,))
        num_targets = 8 
        target_indices = torch.randperm(self.seq_len // 2)[:num_targets]
        target_indices = torch.sort(target_indices)[0]
        
        input_seq = data.clone()
        target_seq = torch.full((self.seq_len,), self.vocab_size - 1)
        
        start_paste = self.seq_len // 2
        target_seq[start_paste:start_paste + num_targets] = input_seq[target_indices]
        
        return input_seq, target_seq

def train_copy_task():
    config = MambaConfig(
        d_model=64,
        n_layers=2,
        vocab_size=16,
        d_state=16,
        expand=2
    )
    
    model = Mamba(config)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    dataset = SelectiveCopyDataset(16, 64, 1000)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    print("Starting Copy Task Training...")
    
    for epoch in range(5):
        total_loss = 0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = nn.functional.cross_entropy(logits.view(-1, 16), batch_y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")

if __name__ == "__main__":
    train_copy_task()
    