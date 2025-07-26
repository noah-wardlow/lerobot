#!/usr/bin/env python3

"""
Train pi0 model on Berkeley Autolab UR5 dataset with GPU.
This script is designed for machines with larger GPU memory.

Usage: PYTHONPATH=src python train_berkeley_pi0_gpu.py
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run pi0 fine-tuning with Berkeley UR5 dataset on GPU."""
    
    # Dataset from your load_berkeley_ur5.py
    repo_id = "IPEC-COMMUNITY/berkeley_autolab_ur5_lerobot"
    
    # Training command for GPU with larger memory
    cmd = [
        sys.executable, "-m", "lerobot.scripts.train",
        f"--dataset.repo_id={repo_id}",
        "--policy.path=lerobot/pi0",  # Load pretrained pi0 model
        "--policy.repo_id=my_berkeley_ur5_pi0",
        "--steps=10000",  # Full training steps
        "--batch_size=4",  # Larger batch size for GPU
        "--eval_freq=1000",
        "--save_freq=1000",
        "--wandb.enable=true",
        "--wandb.project=berkeley_ur5_pi0_gpu"
    ]
    
    print("Pi0 Training on Berkeley UR5 Dataset (GPU)")
    print("=" * 50)
    print(f"Dataset: {repo_id}")
    print("Model: pi0 (fine-tuned from lerobot/pi0)")
    print("Device: GPU (CUDA/MPS)")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 50)
    
    try:
        # Run the training
        result = subprocess.run(cmd, check=True, cwd=Path(__file__).parent)
        print("\n✅ Pi0 training completed successfully!")
        print("Check outputs/train/ directory for checkpoints and logs.")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with exit code {e.returncode}")
        print("This is likely due to insufficient GPU memory.")
        print("Try reducing batch_size or using a machine with more GPU memory.")
        return False
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        return False
    
    return True

if __name__ == "__main__":
    print("🚀 Starting pi0 fine-tuning on Berkeley UR5 dataset...")
    success = main()
    if not success:
        sys.exit(1)