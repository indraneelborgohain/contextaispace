"""Helper functions for loading encoder-decoder models."""

import os
import sys
import subprocess
import torch
from pathlib import Path
from .decoder import Transformer, ModelConfig


def download_gptoss_weights(weights_dir="architecture/open-gpt-oss/weights"):
    """Download GPT-OSS weights if not present."""
    weights_path = Path(weights_dir)
    
    # Check if weights already exist
    if weights_path.exists() and (weights_path / "model.pt").exists():
        print(f"✓ Weights already exist at {weights_dir}")
        return True
    
    print(f"\n{'='*60}")
    print("GPT-OSS WEIGHTS NOT FOUND - DOWNLOADING")
    print(f"{'='*60}")
    
    # Create directory structure
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if download script exists
    download_script = Path("architecture/open-gpt-oss/download_weights.py")
    
    if download_script.exists():
        print(f"Running download script: {download_script}")
        try:
            result = subprocess.run(
                [sys.executable, str(download_script)],
                check=True,
                capture_output=True,
                text=True
            )
            print(result.stdout)
            print("✓ Weights downloaded successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Download script failed: {e}")
            print(e.stderr)
            return False
    else:
        print(f"⚠️  Download script not found at {download_script}")
        print(f"\nPlease download GPT-OSS weights manually:")
        print(f"  1. Clone the GPT-OSS repository")
        print(f"  2. Run: python architecture/open-gpt-oss/download_weights.py")
        print(f"  Or provide your own pretrained weights in {weights_dir}/model.pt")
        return False


def load_decoder(vocab_size, device, encoder_hidden_size=None, load_pretrained=True, weights_dir="architecture/open-gpt-oss/weights"):
    """
    Create decoder with optional cross-attention to encoder.
    Optionally loads pretrained GPT-OSS weights (downloads if not present).
    
    Args:
        vocab_size: Vocabulary size for the decoder
        device: torch device
        encoder_hidden_size: If provided, enables cross-attention to encoder (e.g., 1024 for BERT-large)
        load_pretrained: Whether to load pretrained GPT-OSS weights (default: True)
        weights_dir: Path to GPT-OSS weights directory
    
    Returns:
        Transformer: Decoder model
    """
    print(f"\n{'='*60}")
    print("Creating Decoder")
    print(f"{'='*60}")
    if encoder_hidden_size:
        print(f"Cross-attention enabled (encoder hidden size: {encoder_hidden_size})")
    
    # Create decoder config using defaults from ModelConfig
    decoder_config = ModelConfig(vocab_size=vocab_size)
    
    print(f"\nCreating decoder with config...")
    print(f"  Layers: {decoder_config.num_hidden_layers}")
    print(f"  Hidden size: {decoder_config.hidden_size}")
    print(f"  Intermediate size: {decoder_config.intermediate_size}")
    print(f"  Num experts: {decoder_config.num_experts}")
    print(f"  Experts per token: {decoder_config.experts_per_token}")
    print(f"  Attention heads: {decoder_config.num_attention_heads}")
    print(f"  KV heads: {decoder_config.num_key_value_heads}")
    
    decoder = Transformer(decoder_config, encoder_hidden_size=encoder_hidden_size, device=device)
    
    # Load pretrained GPT-OSS weights if requested
    if load_pretrained:
        # Try to download weights if they don't exist
        weights_available = download_gptoss_weights(weights_dir)
        
        if weights_available and os.path.exists(weights_dir):
            print(f"\nLoading pretrained GPT-OSS weights from {weights_dir}...")
            try:
                # Load the checkpoint
                checkpoint_path = os.path.join(weights_dir, "model.pt")
                if os.path.exists(checkpoint_path):
                    checkpoint = torch.load(checkpoint_path, map_location=device)
                    
                    # Load compatible weights (skip cross-attention layers if encoder_hidden_size is provided)
                    missing_keys, unexpected_keys = decoder.load_state_dict(checkpoint, strict=False)
                    
                    if missing_keys:
                        print(f"  Missing keys (newly added layers): {len(missing_keys)}")
                        if encoder_hidden_size:
                            print(f"    (Expected - cross-attention layers are new)")
                    
                    if unexpected_keys:
                        print(f"  Unexpected keys: {len(unexpected_keys)}")
                    
                    print(f"✓ Pretrained weights loaded successfully!")
                else:
                    print(f"⚠️  Checkpoint file not found at {checkpoint_path}")
                    print(f"  Initializing with random weights...")
            except Exception as e:
                print(f"⚠️  Error loading pretrained weights: {e}")
                print(f"  Initializing with random weights...")
        else:
            print(f"\nInitializing with random weights (weights not available)")
    else:
        print(f"\nInitializing with random weights (load_pretrained=False)")
    
    print(f"\nDecoder parameters: {sum(p.numel() for p in decoder.parameters())/1e6:.1f}M\n")
    
    return decoder

