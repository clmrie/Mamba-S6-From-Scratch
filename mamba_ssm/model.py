# mamba_ssm/model.py
import torch.nn as nn
from .config import MambaConfig
from .block import MambaBlock
from .utils import RMSNorm

class Mamba(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([MambaBlock(config) for _ in range(config.n_layers)])
        self.norm_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            x = x + layer(x)
            
        x = self.norm_f(x)
        logits = self.lm_head(x)
        
        return logits