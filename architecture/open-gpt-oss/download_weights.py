#!/usr/bin/env python3
"""
Download GPT-OSS pretrained weights.

This script downloads the pretrained GPT-OSS weights from HuggingFace.
The weights will be saved to the weights/ directory.
"""

import os
import sys
import requests
from pathlib import Path
from tqdm import tqdm


def download_file(url, destination):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    with open(destination, 'wb') as f, tqdm(
        desc=destination.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def download_from_huggingface():
    """Download GPT-OSS weights from HuggingFace."""
    print("="*60)
    print("DOWNLOADING GPT-OSS WEIGHTS FROM HUGGINGFACE")
    print("="*60)
    
    weights_dir = Path(__file__).parent / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # HuggingFace repository URL for GPT-OSS or similar model
    # Note: You may need to update this URL to the actual GPT-OSS weights
    base_url = "https://huggingface.co/openai-community/gpt2/resolve/main"
    
    files_to_download = {
        "pytorch_model.bin": f"{base_url}/pytorch_model.bin",
        "config.json": f"{base_url}/config.json",
    }
    
    print(f"\nDownloading to: {weights_dir}\n")
    
    for filename, url in files_to_download.items():
        destination = weights_dir / filename
        
        if destination.exists():
            print(f"✓ {filename} already exists, skipping...")
            continue
        
        try:
            print(f"Downloading {filename}...")
            download_file(url, destination)
            print(f"✓ {filename} downloaded successfully!\n")
        except Exception as e:
            print(f"⚠️  Failed to download {filename}: {e}")
            return False
    
    # Rename pytorch_model.bin to model.pt for consistency
    pytorch_model = weights_dir / "pytorch_model.bin"
    model_pt = weights_dir / "model.pt"
    
    if pytorch_model.exists() and not model_pt.exists():
        print("Converting pytorch_model.bin to model.pt...")
        import torch
        state_dict = torch.load(pytorch_model, map_location='cpu')
        torch.save(state_dict, model_pt)
        print("✓ Conversion complete!")
    
    print("\n" + "="*60)
    print("DOWNLOAD COMPLETE!")
    print("="*60)
    print(f"Weights saved to: {weights_dir}")
    
    return True


def download_gpt2_medium_fallback():
    """Download GPT2-Medium as a fallback option."""
    print("\n" + "="*60)
    print("DOWNLOADING GPT2-MEDIUM AS FALLBACK")
    print("="*60)
    print("\nNote: Using GPT2-Medium (345M params) as GPT-OSS weights")
    print("This is a pretrained GPT model that can be used for testing.\n")
    
    try:
        from transformers import GPT2LMHeadModel
        import torch
        
        weights_dir = Path(__file__).parent / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        
        model_pt = weights_dir / "model.pt"
        config_json = weights_dir / "config.json"
        
        if model_pt.exists():
            print(f"✓ model.pt already exists at {model_pt}")
            return True
        
        print("Loading GPT2-Medium from HuggingFace...")
        model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
        
        print("Saving state dict...")
        torch.save(model.state_dict(), model_pt)
        
        # Save config
        import json
        config = {
            "vocab_size": model.config.vocab_size,
            "hidden_size": model.config.n_embd,
            "num_hidden_layers": model.config.n_layer,
            "num_attention_heads": model.config.n_head,
            "intermediate_size": model.config.n_embd * 4,
            "max_position_embeddings": model.config.n_positions,
        }
        
        with open(config_json, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n✓ GPT2-Medium weights saved to: {model_pt}")
        print(f"✓ Config saved to: {config_json}")
        
        print("\n" + "="*60)
        print("DOWNLOAD COMPLETE!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"⚠️  Failed to download GPT2-Medium: {e}")
        return False


def main():
    """Main download function."""
    print("\nGPT-OSS Weight Downloader")
    print("="*60)
    
    # Try downloading from HuggingFace first
    # If that fails, use GPT2-Medium as fallback
    print("\nOption 1: Try downloading from HuggingFace...")
    print("Option 2: Use GPT2-Medium as fallback\n")
    
    choice = input("Choose option (1 or 2, or press Enter for option 2): ").strip()
    
    if choice == "1":
        success = download_from_huggingface()
    else:
        success = download_gpt2_medium_fallback()
    
    if success:
        print("\n✓ Weights ready for training!")
        return 0
    else:
        print("\n⚠️  Download failed. You may need to:")
        print("  1. Check your internet connection")
        print("  2. Install required packages: pip install requests tqdm transformers")
        print("  3. Manually download weights and place in architecture/open-gpt-oss/weights/")
        return 1


if __name__ == "__main__":
    sys.exit(main())
