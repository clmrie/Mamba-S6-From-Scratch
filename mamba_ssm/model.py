# mamba_ssm/model.py
import torch
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

    def forward(self, input_ids, inference_params=None):
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            x = x + layer(x, inference_params)
            
        x = self.norm_f(x)
        logits = self.lm_head(x)
        
        return logits

    def generate(self, input_ids, max_new_tokens=50):
        self.eval()
        inference_params = {}
        
        current_input = input_ids
        
        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits = self(current_input, inference_params)
            
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
            current_input = next_token
            
        return input_ids