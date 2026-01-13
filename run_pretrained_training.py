#!/usr/bin/env python3
"""
Example usage of train_with_pretrained.py

This script shows how to fine-tune your trained encoder-decoder with new learning rates.
Before running, make sure you have a trained encoder-decoder checkpoint.
"""

import os
import subprocess
import sys

def check_requirements():
    """Check if prerequisites are met"""
    print("="*60)
    print("Checking Prerequisites")
    print("="*60)
    
    # Check for encoder-decoder checkpoint
    checkpoint_dirs = [
        "model_msmarco",
        "model_squad",
        "model_stories",
        "models",
        "checkpoints"
    ]
    
    checkpoint_found = None
    for dir_name in checkpoint_dirs:
        if os.path.exists(dir_name):
            checkpoints = [f for f in os.listdir(dir_name) if f.endswith('.pt')]
            if checkpoints:
                checkpoint_found = os.path.join(dir_name, checkpoints[-1])
                print(f"✓ Found checkpoint: {checkpoint_found}")
                break
    
    if not checkpoint_found:
        print("✗ No checkpoint found!")
        print("  Please train a model first using one of:")
        print("    - python train_msmarco.py")
        print("    - python train_squad.py")
        print("    - python train_encoder_decoder_stories.py")
        return None
    
    print("="*60 + "\n")
    
    return checkpoint_found


def run_training_example(encoder_checkpoint, gptoss_weights):
    """Run the training with pretrained weights"""
    
    cmd = [
        sys.executable,
        "train_with_pretrained.py",
        "--encoder_checkpoint", encoder_checkpoint,
        "--gptoss_weights", gptoss_weights,
        "--out_dir", "model_pretrained_gptoss",
        # Learning rates (key parameters!)
        "--encoder_lr", "1e-5",
        "--decoder_lr", "1e-6",
        "--cross_attn_lr", "3e-4",
        # Training params
        "--batch_size", "4",
        "--max_iters", "5000",
        "--max_context_lecheckpoint):
    """Run the fine-tuning"""
    
    cmd = [
        sys.executable,
        "train_with_pretrained.py",
        "--checkpoint", checkpoint,
        "--out_dir", "model_finetuned",
        # Learning rates (lower than initial training)
        "--encoder_lr", "1e-5",
        "--decoder_lr", "1e-5
    
    print("Running command:")
    print(" ".join(cmd))
    print("\n" + "="*60 + "\n")
    
    subprocess.run(cmd)


def main():
    result = check_requirements()
    
    checkpoint = check_requirements()
    
    if checkpoint is None:
        print("\n❌ Prerequisites not met. Please address the issues above.")
        return
    
    # Ask user confirmation
    print("Configuration:")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Learning rates:")
    print(f"    - Encoder: 1e-5")
    print(f"    - Decoder: 1e-5")
    print()
    
    response = input("Start fine-tuning? [y/N]: ")
    if response.lower() == 'y':
        run_training_example(checkpoint)
    else:
        print("Training cancelled.")
        print("\nTo run manually:")
        print(f"python train_with_pretrained.py \\")
        print(f"    --checkpoint {checkpoint} \\")
        print(f"    --encoder_lr 1e-5 --decoder_lr 1e-5
if __name__ == "__main__":
    main()
