import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.s6 import S6

class BiDirectionalMambaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.in_proj = nn.Linear(config.d_model, config.d_inner * 2, bias=config.bias)
        self.conv1d = nn.Conv1d(
            in_channels=config.d_inner,
            out_channels=config.d_inner,
            bias=config.conv_bias,
            kernel_size=config.d_conv,
            groups=config.d_inner,
            padding=config.d_conv - 1,
        )
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)
        
        self.forward_s6 = S6(config)
        self.backward_s6 = S6(config)

    def forward(self, x):
        x_and_res = self.in_proj(x)
        (x, res) = x_and_res.split(split_size=[self.config.d_inner, self.config.d_inner], dim=-1)

        x = x.permute(0, 2, 1)
        x = self.conv1d(x)[:, :, :x.shape[-1]]
        x = x.permute(0, 2, 1)
        x = F.silu(x)

        out_fwd = self.forward_s6(x)
        
        x_rev = torch.flip(x, dims=[1])
        out_rev = self.backward_s6(x_rev)
        out_rev = torch.flip(out_rev, dims=[1])
        
        x = (out_fwd + out_rev) / 2
        
        res = F.silu(res)
        x = x * res
        return self.out_proj(x)