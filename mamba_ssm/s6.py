# mamba_ssm/s6.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# JIT compiles this loop to C++ for speed (replaces torch.func)
@torch.jit.script
def scan_jit(x, dA, dB, C, D):
    B, L, D_model = x.shape
    N = dA.shape[-1]
    
    # Force float32 for stability during accumulation
    h = torch.zeros(B, D_model, N, device=x.device, dtype=torch.float32)
    y = torch.zeros(B, L, D_model, device=x.device, dtype=x.dtype)
    
    dBx = dB * x.unsqueeze(-1)
    
    for t in range(L):
        # Recurrence: h_t = dA * h_{t-1} + dB * x
        h = dA[:, t] * h + dBx[:, t]
        
        # Output: y_t = h_t * C + x * D
        # Contract over state dimension N
        C_t = C[:, t].unsqueeze(1) # [B, 1, N]
        y_val = torch.sum(h * C_t, dim=-1) # [B, D]
        
        y[:, t] = y_val.to(y.dtype) + x[:, t] * D
        
    return y

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

        # S4D-Real Initialization for stability
        dt_min = 0.001
        dt_max = 0.1
        inv_dt_min = torch.log(torch.exp(torch.tensor(dt_min)) - 1)
        inv_dt_max = torch.log(torch.exp(torch.tensor(dt_max)) - 1)
        with torch.no_grad():
            self.dt_proj.bias.uniform_(inv_dt_min, inv_dt_max)
            nn.init.uniform_(self.dt_proj.weight, -0.1, 0.1)

    def forward(self, x, inference_params=None):
        b, l, d = x.shape
        
        x_proj = self.x_proj(x)
        dt_x, B, C = torch.split(x_proj, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        dt = F.softplus(self.dt_proj(dt_x))
        A = -torch.exp(self.A_log) # Enforce negative A

        dA = torch.exp(torch.einsum('b l d, d n -> b l d n', dt, A))
        dB = torch.einsum('b l d, b l n -> b l d n', dt, B)
        
        if inference_params is not None:
            return self.step(x, dA, dB, C, inference_params)

        # Use the JIT compiled scan
        y = scan_jit(x, dA, dB, C, self.D)
        
        return y

    def step(self, x, dA, dB, C, inference_params):
        h_prev = inference_params.get('ssm_state', torch.zeros(x.shape[0], self.d_inner, self.d_state, device=x.device))
        h = dA[:, 0] * h_prev + dB[:, 0] * x[:, 0].unsqueeze(-1)
        inference_params['ssm_state'] = h
        y = torch.einsum('b d n, b n -> b d', h, C[:, 0])
        y = y + x[:, 0] * self.D
        return y.unsqueeze(1)