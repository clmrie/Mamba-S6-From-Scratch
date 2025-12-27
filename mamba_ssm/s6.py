
# mamba_ssm/s6.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class S6(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_inner = config.d_inner
        self.d_state = config.d_state
        self.dt_rank = config.dt_rank
        self.d_model = config.d_model

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
    def forward(self, x):
        b, l, d = x.shape
        
        x_proj = self.x_proj(x)
        dt_x, B, C = torch.split(x_proj, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        dt = F.softplus(self.dt_proj(dt_x))
        A = -torch.exp(self.A_log) 

        dA = torch.exp(torch.einsum('b l d, d n -> b l d n', dt, A))
        dB = torch.einsum('b l d, b l n -> b l d n', dt, B)

        h = torch.zeros(b, self.d_inner, self.d_state, device=x.device)
        ys = []
        
        for t in range(l):
            h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)
            y = torch.einsum('b d n, b n -> b d', h, C[:, t])
            ys.append(y)
            
        y = torch.stack(ys, dim=1) 
        y = y + x * self.D
        
        return y
    