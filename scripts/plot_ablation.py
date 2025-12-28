
import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    bidi_file = "results_mnist/mnist_scores.csv"
    causal_file = "results_mnist/mnist_causal_scores.csv"
    
    if not os.path.exists(bidi_file) or not os.path.exists(causal_file):
        print("Error: Missing score files. Run both training scripts first.")
        return

    df_bidi = pd.read_csv(bidi_file)
    df_causal = pd.read_csv(causal_file)

    plt.figure(figsize=(8, 6))
    
    plt.plot(df_bidi['epoch'], df_bidi['test_accuracy'], 
             marker='o', label='Bi-Directional (Ours)', color='#1f77b4', linewidth=2)
    
    plt.plot(df_causal['epoch'], df_causal['test_accuracy'], 
             marker='x', label='Causal (Baseline)', color='#ff7f0e', linewidth=2, linestyle='--')

    plt.title('Ablation Study: Effect of Bi-Directional Scan', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig("ablation_study.png", dpi=300)
    print("Saved ablation_study.png")
    plt.show()

if __name__ == "__main__":
    main()
    