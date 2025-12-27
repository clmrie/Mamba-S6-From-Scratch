import torch
import torch.nn as nn
from mamba_ssm.utils import RMSNorm
from .block import BiDirectionalMambaBlock

class VisionMamba(nn.Module):
    def __init__(self, config, num_classes=10, channels=1):
        super().__init__()
        self.config = config
        
        self.patch_embed = nn.Conv2d(channels, config.d_model, kernel_size=4, stride=4)
        
        self.layers = nn.ModuleList([BiDirectionalMambaBlock(config) for _ in range(config.n_layers)])
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