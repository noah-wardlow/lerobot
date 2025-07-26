#!/usr/bin/env python3

"""
Train pi0 model on Berkeley Autolab UR5 dataset.
Usage: PYTHONPATH=src python train_berkeley_ur5_pi0.py
"""

import subprocess
import sys
from pathlib import Path

def run_training():
    """Run pi0 training with Berkeley UR5 dataset."""
    
    # Dataset from your load_berkeley_ur5.py
    repo_id = "IPEC-COMMUNITY/berkeley_autolab_ur5_lerobot"
    
    # Training command using LeRobot's train script
    cmd = [
        sys.executable, "-m", "lerobot.scripts.train",
        f"--dataset.repo_id={repo_id}",
        "--policy.type=pi0",
        "--policy.n_obs_steps=1",
        "--policy.chunk_size=50",
        "--policy.n_action_steps=50",
        "--steps=100000",
        "--batch_size=8",
        "--eval_freq=10000",
        "--save_freq=10000",
        "--wandb.enable=true",
        f"--wandb.project=berkeley_ur5_pi0"
    ]
    
    print("Starting pi0 training on Berkeley UR5 dataset...")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 80)
    
    try:
        # Run the training
        result = subprocess.run(cmd, check=True, cwd=Path(__file__).parent)
        print("\n✓ Training completed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Training failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")
        return False
    
    return True

def run_pretrained_finetuning():
    """Alternative: Fine-tune from pretrained pi0 model."""
    
    repo_id = "IPEC-COMMUNITY/berkeley_autolab_ur5_lerobot"
    
    cmd = [
        sys.executable, "-m", "lerobot.scripts.train",
        f"--dataset.repo_id={repo_id}",
        "--policy.path=lerobot/pi0",  # Load pretrained model
        "--steps=50000",  # Fewer steps for finetuning
        "--batch_size=8",
        "--eval_freq=5000",
        "--save_freq=5000",
        "--wandb.enable=true",
        f"--wandb.project=berkeley_ur5_pi0_finetune"
    ]
    
    print("Starting pi0 fine-tuning on Berkeley UR5 dataset...")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 80)
    
    try:
        result = subprocess.run(cmd, check=True, cwd=Path(__file__).parent)
        print("\n✓ Fine-tuning completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Fine-tuning failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n⚠ Fine-tuning interrupted by user")
        return False

if __name__ == "__main__":
    print("Berkeley UR5 + pi0 Training Options:")
    print("1. Train from scratch")
    print("2. Fine-tune from pretrained pi0 model (recommended)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        success = run_training()
    elif choice == "2":
        success = run_pretrained_finetuning()
    else:
        print("Invalid choice. Please run again and select 1 or 2.")
        sys.exit(1)
    
    if success:
        print("\n🎉 Training/fine-tuning completed!")
        print("Check the outputs/train/ directory for checkpoints and logs.")
    else:
        sys.exit(1)