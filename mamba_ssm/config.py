# mamba_ssm/config.py
from dataclasses import dataclass

@dataclass
class MambaConfig:
    d_model: int = 512
    n_layers: int = 24
    vocab_size: int = 50257
    d_state: int = 16
    expand: int = 2
    dt_rank: int = 'auto'
    d_conv: int = 4
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False

    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        if self.dt_rank == 'auto':
            from math import ceil
            self.dt_rank = ceil(self.d_model / 16)