

import torchvision
import os
import sys

# Force output to show immediately
sys.stdout.reconfigure(line_buffering=True)

print("--- CHECKING DATA ---")
data_path = './data'

if os.path.exists(data_path):
    print(f"Directory {data_path} exists.")
    print("Files found:", os.listdir(data_path))
else:
    print(f"Directory {data_path} does not exist. Creating...")

print("--- STARTING DOWNLOAD ---")
try:
    # Download Train
    print("Downloading Train Set...")
    torchvision.datasets.CIFAR10(root=data_path, train=True, download=True)
    
    # Download Test
    print("Downloading Test Set...")
    torchvision.datasets.CIFAR10(root=data_path, train=False, download=True)
    
    print("--- SUCCESS: Download Complete ---")
    
    # Verify file sizes
    print("Verifying files in ./data/cifar-10-batches-py/:")
    cifar_dir = os.path.join(data_path, 'cifar-10-batches-py')
    if os.path.exists(cifar_dir):
        for f in os.listdir(cifar_dir):
            size = os.path.getsize(os.path.join(cifar_dir, f)) / (1024 * 1024)
            print(f"  - {f}: {size:.2f} MB")
    else:
        print("WARNING: Extraction folder not found.")

except Exception as e:
    print(f"--- FAILURE: {e} ---")
    