# mamba_ssm/block.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .s6 import S6

class MambaBlock(nn.Module):
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
        self.x_proj = S6(config)
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

    def forward(self, x, inference_params=None):
        x_and_res = self.in_proj(x)
        (x, res) = x_and_res.split(split_size=[self.config.d_inner, self.config.d_inner], dim=-1)

        x = x.permute(0, 2, 1)
        
        if inference_params is not None:
            conv_state = inference_params.get('conv_state', torch.zeros(x.shape[0], self.config.d_inner, self.config.d_conv, device=x.device))
            conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
            conv_state[:, :, -1] = x[:, :, 0]
            inference_params['conv_state'] = conv_state
            x = torch.sum(conv_state * self.conv1d.weight.squeeze(), dim=-1)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = x.unsqueeze(-1)
        else:
            x = self.conv1d(x)[:, :, :x.shape[-1]]

        x = x.permute(0, 2, 1)
        x = F.silu(x)

        x = self.x_proj(x, inference_params)
        
        res = F.silu(res)
        x = x * res
        
        return self.out_proj(x)