import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import numpy as np

# 导入你的 Mamba 模型
from mamba_ssm.config import MambaConfig
from mamba_ssm.model import Mamba

# --- 1. 定义一个简单的 Transformer Baseline (PyTorch 原生) ---
class SimpleTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 使用 PyTorch 官方的 TransformerEncoderLayer 模拟 GPT 结构
        # 为了公平，不使用 FlashAttention，只用标准的 Attention
        encoder_layer = nn.TransformerEncoderLayer(d_model=config.d_model, nhead=4, dim_feedforward=config.d_model*4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)
        
    def forward(self, x):
        x = self.embedding(x)
        # 生成 causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(x.device)
        out = self.transformer(x, mask=mask, is_causal=True)
        return self.lm_head(out)

# --- 2. 测速函数 ---
def measure_latency(model, model_type, seq_lengths, device="cpu", generate_tokens=10):
    times = []
    
    print(f"Testing {model_type} on {device}...")
    
    for seq_len in seq_lengths:
        # 伪造不同长度的输入 (Batch size = 1)
        input_ids = torch.randint(0, 100, (1, seq_len)).to(device)
        
        # 预热 (Warmup) - 让 CPU/GPU 缓存准备好
        with torch.no_grad():
            if model_type == "Mamba":
                _ = model(input_ids)
            else:
                _ = model(input_ids)
        
        start_time = time.time()
        
        # 模拟生成过程
        # 注意：这里我们主要测试 "Prefill + Decode 1 step" 或者 "Decode loop"
        # 为了展示趋势，我们简单测试：给定长 Context，做一次前向传播的时间 (Next Token Prediction Time)
        with torch.no_grad():
            for _ in range(generate_tokens):
                if model_type == "Mamba":
                    # Mamba 有 inference_params 缓存，推理应该是 O(1)
                    # 但为了简单对比单纯的前向传播计算量，
                    # Transformer 在没有 KV Cache 时是 O(N^2)，Mamba 是 O(N) (Scan)
                    # 如果要对比真正的生成速度，需要写完整的 generate loop。
                    # 这里我们用最简单的方式：输入长度为 N，计算一次 Forward
                    _ = model(input_ids) 
                else:
                    # Transformer 输入长度 N，计算一次 Forward (模拟没有 KV Cache 的情况，或者 Cache 满了的情况)
                    _ = model(input_ids)
        
        end_time = time.time()
        
        # 计算平均每个 token 的耗时 (ms)
        # 注意：这里我们测试的是 "处理长序列的能力"
        avg_time = (end_time - start_time) / generate_tokens * 1000 
        times.append(avg_time)
        print(f"Length {seq_len}: {avg_time:.2f} ms")
        
    return times

# --- 3. 主程序 ---
def run_experiment():
    # 配置
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    seq_lengths = [2000,4000,6000, 8000,10000, 12000, 14000,16000,18000] # 序列长度逐渐增加
    
    config = MambaConfig(d_model=64, n_layers=2, vocab_size=200) # 小模型，只测速度
    
    # 初始化模型
    mamba_model = Mamba(config).to(device).eval()
    transformer_model = SimpleTransformer(config).to(device).eval()
    
    # 运行测试
    # 注意：这里为了凸显 Transformer 的 O(N^2) 瓶颈，我们测试的是 "Full Forward Pass"
    # (即每次都把整个序列丢进去算，这最能体现 Transformer Attention 的代价)
    mamba_times = measure_latency(mamba_model, "Mamba", seq_lengths, device)
    transformer_times = measure_latency(transformer_model, "Transformer", seq_lengths, device)
    
    # --- 4. 画图 ---
    plt.figure(figsize=(10, 6))
    plt.plot(seq_lengths, transformer_times, 'r-o', label='Transformer (Attention $O(N^2)$)')
    plt.plot(seq_lengths, mamba_times, 'g-o', label='Mamba (Scan $O(N)$)')
    
    plt.xlabel('Sequence Length (Context Size)')
    plt.ylabel('Inference Time (ms)')
    plt.title('Inference Speed Comparison: Mamba vs Transformer')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('inference_speed_comparison.png')
    print("Picture saved as inference_speed_comparison.png")

if __name__ == "__main__":
    run_experiment()