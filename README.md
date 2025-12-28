
# From Sequence to Spatial: Mamba S6 From Scratch

This repository contains a pure PyTorch reimplementation of the **Mamba** architecture (Selective State Space Models) and a novel extension to Computer Vision tasks. The project focuses on demystifying the Selective Scan (S6) mechanism by implementing it from scratch without custom CUDA kernels, and adapting its inherently causal nature for bi-directional spatial understanding.

## 🚀 Key Contributions

* **S6 Implementation**: A clean, educational implementation of the Selective Scan algorithm using `torch.jit.script` for hardware-efficient operation fusion in pure PyTorch.
* **Bi-Directional Vision Mamba**: A custom block designed for 2D data that fuses forward and backward scans, solving the receptive field limitations of causal models on images.
* **Scientific Validation**: Rigorous ablation studies on MNIST and character-level language modeling on TinyShakespeare.

---

## 📊 Experimental Results

### 1. Superior Convergence (Mamba vs. RNN)
We compared our Vision Mamba implementation against a standard Vanilla RNN baseline on pixel-sequence classification. Mamba demonstrates significantly faster convergence and stabilizes at a lower loss floor.

![Mamba vs RNN](outputs/figures/mamba_vs_rnn.png)

### 2. Ablation Study: The Necessity of Bi-Directionality
Standard Mamba is causal (left-to-right), which restricts a pixel's context to its predecessors. Our **Bi-Directional Extension** removes this bottleneck, resulting in higher accuracy and improved training stability compared to the causal baseline.

| Model | Peak Test Accuracy |
| :--- | :--- |
| **Vanilla RNN Baseline** | 94.40% |
| **Causal Mamba (Baseline)** | 97.00% |
| **Bi-Directional Vision Mamba (Ours)** | **97.93%** |

![Ablation Study](outputs/figures/ablation_study.png)

### 3. Interpretability: Visualizing Selection ($\Delta$)
A key feature of Mamba is the input-dependent selection mechanism. Below, we visualize the magnitude of the discretization parameter $\Delta$ as the model scans a digit "7". The spikes correspond to the structural strokes of the digit, confirming that the model learns to "open its gates" only for relevant information.

![Delta Visualization](outputs/figures/delta_visualization.png)

---

## 📂 Project Structure

```text
.
├── mamba_ssm/              # Core S6 implementation and Mamba Blocks
│   ├── s6.py               # The Selective Scan algorithm (JIT compiled)
│   └── block.py            # Mamba Block architecture
├── vision/                 # Vision-specific adaptations
│   └── model.py            # Bi-Directional Vision Mamba Model
├── scripts/                # Visualization and plotting tools
├── outputs/                # Generated figures and logs
├── train_mnist.py          # Main training script (Bi-Directional)
├── train_mnist_causal.py   # Ablation script (Unidirectional)
├── train_rnn_baseline.py   # Baseline RNN training script
└── train_shakespeare.py    # Language modeling verification script

```

---

## 🛠️ Usage

### Installation

Clone the repository and install the necessary dependencies:

```bash
git clone [https://github.com/clmrie/Mamba-S6-From-Scratch](https://github.com/clmrie/Mamba-S6-From-Scratch)
cd Mamba-S6-From-Scratch
pip install torch torchvision matplotlib pandas

```

### Training

**Train Bi-Directional Vision Mamba (Best Model):**

```bash
python train_mnist.py

```

**Train Vanilla RNN Baseline:**

```bash
python train_rnn_baseline.py

```

**Train Causal Mamba (Ablation):**

```bash
python train_mnist_causal.py

```

### Reproducing Figures

To generate the visualization plots shown above:

```bash
python scripts/make_comparison_plot.py  # Generates convergence plot
python scripts/plot_ablation.py         # Generates ablation study
python scripts/visualize_delta.py       # Generates mechanism visualization

```

---

## 📜 References

* **Paper:** Gu, A., & Dao, T. (2023). [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752).
* **Original Implementation:** [state-spaces/mamba](https://github.com/state-spaces/mamba)

```

