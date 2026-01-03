import os
import sys
sys.stdout.reconfigure(line_buffering=True)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())

from mamba_ssm.config import MambaConfig
from vision.model import VisionMamba


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class VanillaRNN(nn.Module):
    """
    MNIST -> 4x4 patches -> 49 tokens of dim 16.
    RNN backbone, last token -> classifier.
    """
    def __init__(self, hidden_size: int, num_layers: int = 2, patch_size: int = 4):
        super().__init__()
        self.patch_size = patch_size
        self.input_size = patch_size * patch_size  # 16 for 4x4
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(self.hidden_size, 10)

    def forward(self, x):
        B, C, H, W = x.shape
        ps = self.patch_size

        x = x.unfold(2, ps, ps).unfold(3, ps, ps)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(B, -1, self.input_size)

        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


def pick_rnn_hidden_to_match_params(target_params: int, num_layers: int = 2, patch_size: int = 4) -> tuple[int, int]:
    """
    Brute-force hidden size selection to match target_params as closely as possible.
    Returns (best_hidden, best_params).
    """
    best_h = None
    best_p = None
    best_diff = None

    # Reasonable search range for this setup
    for h in range(32, 513):
        tmp = VanillaRNN(hidden_size=h, num_layers=num_layers, patch_size=patch_size)
        p = count_trainable_params(tmp)
        diff = abs(p - target_params)
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_h = h
            best_p = p

    return best_h, best_p


def main():
    torch.manual_seed(0)

    mamba_log_file = "results_mnist/mnist_training_log.csv"
    if not os.path.exists(mamba_log_file):
        print(f"Error: Could not find {mamba_log_file}. Did you run train_mnist.py?")
        return

    print("--- STEP 0: Computing Vision Mamba parameter count ---")
    config = MambaConfig(
        d_model=64,
        n_layers=2,
        vocab_size=0,
        d_state=16,
        expand=2
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mamba_model = VisionMamba(config, num_classes=10, channels=1).to(device)
    mamba_params = count_trainable_params(mamba_model)
    print(f"Vision Mamba params: {mamba_params:,}")

    print("--- STEP 1: Training Vanilla RNN Baseline (parameter-matched) ---")
    BATCH_SIZE = 64
    EPOCHS = 5
    LR = 1e-3
    WEIGHT_DECAY = 0.01
    CLIP_NORM = 1.0

    PATCH_SIZE = 4
    RNN_LAYERS = 2

    best_hidden, rnn_params = pick_rnn_hidden_to_match_params(
        target_params=mamba_params,
        num_layers=RNN_LAYERS,
        patch_size=PATCH_SIZE
    )
    print(f"Chosen RNN hidden_size={best_hidden} to match params.")
    print(f"RNN params: {rnn_params:,}")
    print(f"Absolute diff: {abs(rnn_params - mamba_params):,} params")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

    rnn_model = VanillaRNN(hidden_size=best_hidden, num_layers=RNN_LAYERS, patch_size=PATCH_SIZE).to(device)

    # Match training recipe style to your Mamba run
    optimizer = optim.AdamW(rnn_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    rnn_history = []
    global_step = 0

    for epoch in range(EPOCHS):
        rnn_model.train()
        running_loss = 0.0

        for i, (x, y) in enumerate(trainloader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)
            out = rnn_model(x)
            loss = criterion(out, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(rnn_model.parameters(), CLIP_NORM)
            optimizer.step()

            running_loss += loss.item()

            if i % 10 == 9:
                rnn_history.append({
                    "global_step": global_step,
                    "loss": running_loss / 10.0
                })
                running_loss = 0.0

            global_step += 1

        print(f"RNN Epoch {epoch+1}/{EPOCHS} complete.")

    print("--- STEP 2: Generating Comparison Plot ---")
    df_rnn = pd.DataFrame(rnn_history)
    df_mamba = pd.read_csv(mamba_log_file)

    batches_per_epoch = len(trainloader)
    df_mamba["global_step"] = (df_mamba["epoch"] - 1) * batches_per_epoch + (df_mamba["batch"] - 1)

    plt.figure(figsize=(10, 6))
    plt.plot(
        df_mamba["global_step"],
        df_mamba["loss"],
        label=f"Vision Mamba ({mamba_params/1000:.1f}k params)",
        alpha=0.9
    )
    plt.plot(
        df_rnn["global_step"],
        df_rnn["loss"],
        label=f"Vanilla RNN ({rnn_params/1000:.1f}k params)",
        alpha=0.7,
        linestyle="--"
    )

    plt.title("Convergence: Mamba vs Vanilla RNN on MNIST (parameter-matched)")
    plt.xlabel("Training Steps (batches)")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 2.5)

    plt.savefig("mamba_vs_rnn.png", dpi=300, bbox_inches="tight")
    print("✅ Saved mamba_vs_rnn.png")


if __name__ == "__main__":
    main()
