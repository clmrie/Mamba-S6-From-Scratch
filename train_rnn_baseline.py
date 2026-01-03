import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# Force immediate printing
sys.stdout.reconfigure(line_buffering=True)


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class VanillaRNN(nn.Module):
    """
    Vanilla RNN baseline with MNIST patchification to match the Vision Mamba tokenization:
    - Input: 28x28 image
    - Patch: 4x4 => 7x7 = 49 tokens
    - Token dim: 16
    """
    def __init__(self, hidden_dim: int = 160, num_layers: int = 2, patch_size: int = 4):
        super().__init__()
        self.input_dim = patch_size * patch_size  # 16 for 4x4 patches
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.patch_size = patch_size

        self.rnn = nn.RNN(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bias=True,
            nonlinearity="tanh",
        )
        self.fc = nn.Linear(self.hidden_dim, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 28, 28)
        B, C, H, W = x.shape
        ps = self.patch_size

        # Patchify: (B, C, 7, 7, ps, ps)
        x = x.unfold(2, ps, ps).unfold(3, ps, ps)
        # (B, 7, 7, C, ps, ps)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        # (B, 49, 16)
        x = x.view(B, -1, self.input_dim)

        out, _ = self.rnn(x)           # (B, 49, hidden_dim)
        logits = self.fc(out[:, -1])   # last token => (B, 10)
        return logits


def main():
    # Training hyperparams
    BATCH_SIZE = 64
    EPOCHS = 5
    LR = 1e-3

    # Param-matching knobs
    # hidden_dim=160, num_layers=2 => ~81.6k params (close to an 80k Mamba config)
    HIDDEN_DIM = 160
    NUM_LAYERS = 2
    PATCH_SIZE = 4

    # Repro (optional)
    torch.manual_seed(0)

    # Ensure results folder exists
    RESULTS_DIR = "results_mnist"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    SCORE_FILE = os.path.join(RESULTS_DIR, "rnn_scores.csv")
    LOG_FILE = os.path.join(RESULTS_DIR, "rnn_training_log.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Vanilla RNN Baseline on {device}...")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = VanillaRNN(
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        patch_size=PATCH_SIZE,
    ).to(device)

    nparams = count_trainable_params(model)
    print(f"RNN config: hidden_dim={HIDDEN_DIM}, num_layers={NUM_LAYERS}, patch_size={PATCH_SIZE}")
    print(f"Trainable parameters: {nparams:,}")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # Init Log Files
    with open(LOG_FILE, "w") as f:
        f.write("epoch,batch,loss\n")
    with open(SCORE_FILE, "w") as f:
        f.write("epoch,test_accuracy\n")

    print("Starting RNN Training...")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for i, (x, y) in enumerate(trainloader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 10 == 9:
                avg_loss = running_loss / 10
                with open(LOG_FILE, "a") as f:
                    f.write(f"{epoch+1},{i+1},{avg_loss:.4f}\n")
                running_loss = 0.0

        # Test accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in testloader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                pred = out.argmax(dim=1)
                total += y.size(0)
                correct += (pred == y).sum().item()

        acc = 100.0 * correct / total
        print(f"Epoch {epoch+1} RNN Test Accuracy: {acc:.2f}%")

        with open(SCORE_FILE, "a") as f:
            f.write(f"{epoch+1},{acc:.2f}\n")


if __name__ == "__main__":
    main()
