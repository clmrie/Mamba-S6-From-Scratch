
# train.py
import torch
from mamba_ssm.config import MambaConfig
from mamba_ssm.model import Mamba

def main():
    config = MambaConfig(
        d_model=256,
        n_layers=4,
        vocab_size=1000,
        d_state=16,
        expand=2
    )
    
    model = Mamba(config)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())}")

    input_ids = torch.randint(0, 1000, (2, 64))
    
    logits = model(input_ids)
    print(f"Input Shape: {input_ids.shape}")
    print(f"Output Shape: {logits.shape}") 

if __name__ == "__main__":
    main()
    