#!/usr/bin/env python3
"""
Check available model checkpoints and GPT-OSS weights
"""

import os
from pathlib import Path
from datetime import datetime


def format_size(size_bytes):
    """Format bytes to human readable"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def find_checkpoints(base_dir="."):
    """Find all .pt checkpoint files"""
    checkpoints = []
    
    for root, dirs, files in os.walk(base_dir):
        # Skip .venv and other irrelevant directories
        if '.venv' in root or 'node_modules' in root or '.git' in root:
            continue
        
        for file in files:
            if file.endswith('.pt'):
                filepath = os.path.join(root, file)
                stat = os.stat(filepath)
                checkpoints.append({
                    'path': filepath,
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime)
                })
    
    return sorted(checkpoints, key=lambda x: x['modified'], reverse=True)


def check_gptoss_weights():
    """Check GPT-OSS weights directory"""
    weights_dir = Path("architecture/open-gpt-oss/weights")
    
    if not weights_dir.exists():
        return None, []
    
    files = list(weights_dir.iterdir())
    total_size = sum(f.stat().st_size for f in files if f.is_file())
    
    return total_size, [f.name for f in files if f.is_file()]


def main():
    print("="*70)
    print("Model Checkpoint and Weight Checker")
    print("="*70)
    
    # Check for encoder/decoder checkpoints
    print("\n📦 TRAINED MODEL CHECKPOINTS (.pt files)\n")
    checkpoints = find_checkpoints()
    
    if not checkpoints:
        print("❌ No checkpoint files found!")
        print("\nYou need to train a model first. Try:")
        print("  python train_msmarco.py --max_iters 1000")
        print("  python train_squad.py --max_iters 1000")
        print("  python train_encoder_decoder_stories.py --max_iters 1000")
    else:
        print(f"✅ Found {len(checkpoints)} checkpoint(s):\n")
        for i, ckpt in enumerate(checkpoints[:10], 1):  # Show first 10
            rel_path = ckpt['path'].replace('\\', '/')
            print(f"{i:2d}. {rel_path}")
            print(f"     Size: {format_size(ckpt['size']):<12} Modified: {ckpt['modified']}")
            print()
        
        if len(checkpoints) > 10:
            print(f"... and {len(checkpoints) - 10} more")
    
    # Check GPT-OSS weights
    print("\n" + "="*70)
    print("🤖 GPT-OSS PRETRAINED WEIGHTS (Optional)\n")
    print("NOTE: Your decoder architecture is NOT compatible with GPT-OSS weights.")
    print("The scripts have been updated to work with your existing decoder.\n")
    
    total_size, files = check_gptoss_weights()
    
    if total_size is None:
        print("ℹ️  GPT-OSS weights directory not found (not needed for your architecture)")
    elif not files:
        print("ℹ️  GPT-OSS weights directory exists but is empty (not needed)")
    else:
        print(f"ℹ️  GPT-OSS weights found but won't be used with your decoder")
        print(f"   Directory: architecture/open-gpt-oss/weights")
        print(f"   Total size: {format_size(total_size)}")
    
    # Summary and next steps
    print("\n" + "="*70)
    print("📋 NEXT STEPS\n")
    
    has_checkpoints = len(checkpoints) > 0
    
    if has_checkpoints:
        print("✅ You have everything needed!")
        print("\nRecommended checkpoint (most recent):")
        print(f"  {checkpoints[0]['path']}")
        print("\nRun fine-tuning with new learning rates:")
        print(f"  python run_pretrained_training.py")
        print("\nOr manually:")
        print(f"  python train_with_pretrained.py \\")
        print(f"      --checkpoint {checkpoints[0]['path']} \\")
        print(f"      --encoder_lr 1e-5 --decoder_lr 1e-5")
    
    else:
        print("❌ No trained checkpoints found")
        print("\n1. Train a model first:")
        print("   python train_msmarco.py --max_iters 1000")
        print("\n2. Then run fine-tuning:")
        print("   python run_pretrained_training.py")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
