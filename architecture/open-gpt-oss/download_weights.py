"""Helper functions for loading encoder-decoder models."""

import os
import sys
import subprocess
import torch
from pathlib import Path
from glob import glob
from typing import Optional, Dict, Any

# Assuming these are imported from your local modules
# from .decoder import Transformer, ModelConfig


def load_safetensors(directory: Path) -> Dict[str, torch.Tensor]:
    """Load all SafeTensor files from a directory into a single state dict."""
    try:
        from safetensors import safe_open
    except ImportError:
        raise ImportError("Please install safetensors: pip install safetensors")
    
    state_dict = {}
    safetensor_files = sorted(glob(str(directory / "*.safetensors")))
    
    if not safetensor_files:
        # Check in original/ subdirectory
        safetensor_files = sorted(glob(str(directory / "original" / "*.safetensors")))
    
    if not safetensor_files:
        raise FileNotFoundError(f"No SafeTensor files found in {directory}")
    
    print(f"  Loading {len(safetensor_files)} SafeTensor file(s)...")
    
    for filepath in safetensor_files:
        print(f"    Loading: {Path(filepath).name}")
        with safe_open(filepath, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    
    return state_dict


def map_gptoss_to_decoder(hf_state_dict: Dict[str, torch.Tensor], 
                          decoder_config: Any,
                          encoder_hidden_size: Optional[int] = None) -> Dict[str, torch.Tensor]:
    """
    Map HuggingFace GPT-OSS weight names to your Transformer decoder format.
    
    GPT-OSS uses a structure like:
        model.embed_tokens.weight
        model.layers.{i}.self_attn.q_proj.weight
        model.layers.{i}.self_attn.k_proj.weight
        model.layers.{i}.self_attn.v_proj.weight
        model.layers.{i}.self_attn.o_proj.weight
        model.layers.{i}.mlp.gate_proj.weight (or MoE equivalent)
        model.layers.{i}.mlp.up_proj.weight
        model.layers.{i}.mlp.down_proj.weight
        model.layers.{i}.input_layernorm.weight
        model.layers.{i}.post_attention_layernorm.weight
        model.norm.weight
        lm_head.weight
    
    Adjust the mapping below based on your Transformer class structure.
    """
    
    mapped_dict = {}
    unmapped_keys = []
    
    # Print some sample keys for debugging
    sample_keys = list(hf_state_dict.keys())[:20]
    print(f"\n  Sample HuggingFace keys:")
    for k in sample_keys:
        print(f"    {k}: {hf_state_dict[k].shape}")
    
    # Define key mapping patterns
    # Adjust these based on your actual Transformer class attribute names
    key_mapping = {
        # Embeddings
        "model.embed_tokens.weight": "tok_embeddings.weight",
        "lm_head.weight": "output.weight",
        
        # Final layer norm
        "model.norm.weight": "norm.weight",
        "model.norm.bias": "norm.bias",
        
        # For RoPE (if stored in weights)
        "model.rotary_emb.inv_freq": "rotary_emb.inv_freq",
    }
    
    # Layer-wise mappings (patterns to replace)
    layer_patterns = {
        # Self-attention
        "model.layers.{i}.self_attn.q_proj.weight": "layers.{i}.attention.wq.weight",
        "model.layers.{i}.self_attn.k_proj.weight": "layers.{i}.attention.wk.weight",
        "model.layers.{i}.self_attn.v_proj.weight": "layers.{i}.attention.wv.weight",
        "model.layers.{i}.self_attn.o_proj.weight": "layers.{i}.attention.wo.weight",
        
        # Layer norms
        "model.layers.{i}.input_layernorm.weight": "layers.{i}.attention_norm.weight",
        "model.layers.{i}.post_attention_layernorm.weight": "layers.{i}.ffn_norm.weight",
        
        # MLP (dense FFN - adjust if using MoE)
        "model.layers.{i}.mlp.gate_proj.weight": "layers.{i}.feed_forward.w1.weight",
        "model.layers.{i}.mlp.up_proj.weight": "layers.{i}.feed_forward.w3.weight",
        "model.layers.{i}.mlp.down_proj.weight": "layers.{i}.feed_forward.w2.weight",
        
        # MoE layers (GPT-OSS specific)
        "model.layers.{i}.mlp.router.weight": "layers.{i}.feed_forward.router.weight",
        "model.layers.{i}.mlp.experts.{e}.gate_proj.weight": "layers.{i}.feed_forward.experts.{e}.w1.weight",
        "model.layers.{i}.mlp.experts.{e}.up_proj.weight": "layers.{i}.feed_forward.experts.{e}.w3.weight",
        "model.layers.{i}.mlp.experts.{e}.down_proj.weight": "layers.{i}.feed_forward.experts.{e}.w2.weight",
    }
    
    # Apply direct mappings
    for hf_key, decoder_key in key_mapping.items():
        if hf_key in hf_state_dict:
            mapped_dict[decoder_key] = hf_state_dict[hf_key]
            print(f"  Mapped: {hf_key} -> {decoder_key}")
    
    # Apply layer-wise mappings
    num_layers = decoder_config.num_hidden_layers if hasattr(decoder_config, 'num_hidden_layers') else 32
    num_experts = decoder_config.num_experts if hasattr(decoder_config, 'num_experts') else 8
    
    for layer_idx in range(num_layers):
        for hf_pattern, decoder_pattern in layer_patterns.items():
            if "{e}" in hf_pattern:
                # MoE expert weights
                for expert_idx in range(num_experts):
                    hf_key = hf_pattern.format(i=layer_idx, e=expert_idx)
                    decoder_key = decoder_pattern.format(i=layer_idx, e=expert_idx)
                    if hf_key in hf_state_dict:
                        mapped_dict[decoder_key] = hf_state_dict[hf_key]
            else:
                hf_key = hf_pattern.format(i=layer_idx)
                decoder_key = decoder_pattern.format(i=layer_idx)
                if hf_key in hf_state_dict:
                    mapped_dict[decoder_key] = hf_state_dict[hf_key]
    
    # Track unmapped keys
    mapped_hf_keys = set()
    for hf_key in hf_state_dict.keys():
        found = False
        for decoder_key in mapped_dict.keys():
            # Simple check - you may need more sophisticated matching
            if any(hf_key == k.format(i=i, e=e) 
                   for k in key_mapping.keys() 
                   for i in range(num_layers) 
                   for e in range(num_experts)):
                found = True
                break
        if not found and hf_key not in key_mapping:
            unmapped_keys.append(hf_key)
    
    if unmapped_keys:
        print(f"\n  ⚠️  {len(unmapped_keys)} unmapped HuggingFace keys (sample):")
        for k in unmapped_keys[:10]:
            print(f"      {k}")
        if len(unmapped_keys) > 10:
            print(f"      ... and {len(unmapped_keys) - 10} more")
    
    return mapped_dict


def download_gptoss_weights(weights_dir: str = "architecture/open-gpt-oss/weights",
                            model_name: str = "gpt-oss-20b") -> bool:
    """Download GPT-OSS weights from HuggingFace if not present."""
    weights_path = Path(weights_dir)
    
    # Check if weights already exist (either model.pt or safetensors)
    has_model_pt = (weights_path / "model.pt").exists()
    has_safetensors = bool(glob(str(weights_path / "*.safetensors"))) or \
                      bool(glob(str(weights_path / "original" / "*.safetensors")))
    
    if has_model_pt or has_safetensors:
        print(f"✓ Weights already exist at {weights_dir}")
        return True
    
    print(f"\n{'='*60}")
    print("GPT-OSS WEIGHTS NOT FOUND - DOWNLOADING")
    print(f"{'='*60}")
    
    # Create directory structure
    weights_path.mkdir(parents=True, exist_ok=True)
    
    try:
        from huggingface_hub import snapshot_download
        
        repo_id = f"openai/{model_name}"
        print(f"Downloading from: {repo_id}")
        print(f"Destination: {weights_path}")
        
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=["original/*"],
            local_dir=weights_path,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        
        print("✓ Weights downloaded successfully!")
        return True
        
    except ImportError:
        print("⚠️  huggingface_hub not installed. Install with: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"⚠️  Download failed: {e}")
        print(f"\nPlease download GPT-OSS weights manually:")
        print(f"  huggingface-cli download openai/{model_name} --include 'original/*' --local-dir {weights_dir}")
        return False


def load_decoder(vocab_size: int, 
                 device: torch.device,
                 encoder_hidden_size: Optional[int] = None, 
                 load_pretrained: bool = True, 
                 weights_dir: str = "architecture/open-gpt-oss/weights",
                 model_name: str = "gpt-oss-20b",
                 auto_download: bool = True):
    """
    Create decoder with optional cross-attention to encoder.
    Optionally loads pretrained GPT-OSS weights (downloads if not present).
    
    Args:
        vocab_size: Vocabulary size for the decoder
        device: torch device
        encoder_hidden_size: If provided, enables cross-attention to encoder (e.g., 1024 for BERT-large)
        load_pretrained: Whether to load pretrained GPT-OSS weights (default: True)
        weights_dir: Path to GPT-OSS weights directory
        model_name: HuggingFace model name (gpt-oss-20b or gpt-oss-120b)
        auto_download: Automatically download weights if not present
    
    Returns:
        Transformer: Decoder model
    """
    # Import your local modules (adjust path as needed)
    from .decoder import Transformer, ModelConfig
    
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
        weights_path = Path(weights_dir)
        
        
        if weights_path.exists():
            print(f"\nLoading pretrained GPT-OSS weights from {weights_dir}...")
            
            try:
                # Check for different weight formats
                model_pt_path = weights_path / "model.pt"
                safetensor_files = glob(str(weights_path / "*.safetensors")) or \
                                   glob(str(weights_path / "original" / "*.safetensors"))
                
                if model_pt_path.exists():
                    # Load PyTorch checkpoint directly
                    print(f"  Loading from model.pt...")
                    checkpoint = torch.load(model_pt_path, map_location="cpu")
                    state_dict = checkpoint
                    
                elif safetensor_files:
                    # Load SafeTensors and map keys
                    print(f"  Loading from SafeTensors...")
                    hf_state_dict = load_safetensors(weights_path)
                    
                    # Map HuggingFace keys to your decoder format
                    print(f"\n  Mapping HuggingFace weights to decoder format...")
                    state_dict = map_gptoss_to_decoder(
                        hf_state_dict, 
                        decoder_config,
                        encoder_hidden_size
                    )
                    
                    # Optionally save as model.pt for faster loading next time
                    print(f"\n  Saving converted weights to {model_pt_path}...")
                    torch.save(state_dict, model_pt_path)
                    print(f"  ✓ Saved for faster loading next time")
                    
                else:
                    print(f"⚠️  No weights found at {weights_path}")
                    print(f"  Initializing with random weights...")
                    state_dict = None
                
                if state_dict is not None:
                    # Load compatible weights (skip cross-attention layers if encoder_hidden_size is provided)
                    missing_keys, unexpected_keys = decoder.load_state_dict(state_dict, strict=False)
                    
                    if missing_keys:
                        print(f"\n  Missing keys (newly added/different layers): {len(missing_keys)}")
                        if encoder_hidden_size:
                            cross_attn_missing = [k for k in missing_keys if 'cross' in k.lower()]
                            print(f"    Cross-attention layers (expected): {len(cross_attn_missing)}")
                        # Show sample of other missing keys
                        other_missing = [k for k in missing_keys if 'cross' not in k.lower()][:5]
                        if other_missing:
                            print(f"    Other missing (sample): {other_missing}")
                    
                    if unexpected_keys:
                        print(f"\n  Unexpected keys (not in model): {len(unexpected_keys)}")
                        print(f"    Sample: {unexpected_keys[:5]}")
                    
                    print(f"\n✓ Pretrained weights loaded successfully!")
                    
            except Exception as e:
                print(f"⚠️  Error loading pretrained weights: {e}")
                import traceback
                traceback.print_exc()
                print(f"  Initializing with random weights...")
        else:
            print(f"\nInitializing with random weights (weights directory not found)")
    else:
        print(f"\nInitializing with random weights (load_pretrained=False)")
    
    # Move to device
    decoder = decoder.to(device)
    
    total_params = sum(p.numel() for p in decoder.parameters())
    trainable_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    
    print(f"\nDecoder parameters:")
    print(f"  Total: {total_params/1e9:.2f}B ({total_params/1e6:.1f}M)")
    print(f"  Trainable: {trainable_params/1e9:.2f}B ({trainable_params/1e6:.1f}M)")
    
    return decoder


# Convenience function for quick testing
def load_gptoss_20b(device: torch.device = torch.device("cuda"),
                    weights_dir: str = "./gpt-oss-20b"):
    """Quick loader for GPT-OSS 20B model."""
    return load_decoder(
        vocab_size=200064,  # GPT-OSS tokenizer vocab size
        device=device,
        encoder_hidden_size=None,
        load_pretrained=True,
        weights_dir=weights_dir,
        model_name="gpt-oss-20b",
        auto_download=True
    )


def load_gptoss_120b(device: torch.device = torch.device("cuda"),
                     weights_dir: str = "./gpt-oss-120b"):
    """Quick loader for GPT-OSS 120B model."""
    return load_decoder(
        vocab_size=200064,  # GPT-OSS tokenizer vocab size
        device=device,
        encoder_hidden_size=None,
        load_pretrained=True,
        weights_dir=weights_dir,
        model_name="gpt-oss-120b",
        auto_download=True
    )



    # Test the loader
    import argparse
    
    parser = argparse.ArgumentParser(description="Test GPT-OSS weight loading")
    parser.add_argument("--weights-dir", default="./gpt-oss-20b", help="Path to weights")
    parser.add_argument("--model", default="gpt-oss-20b", choices=["gpt-oss-20b", "gpt-oss-120b"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    print(f"Testing weight loading...")
    print(f"  Weights dir: {args.weights_dir}")
    print(f"  Model: {args.model}")
    print(f"  Device: {args.device}")
    
    # This will fail without your decoder.py, but shows the interface
    try:
        decoder = load_decoder(
            vocab_size=200064,
            device=torch.device(args.device),
            weights_dir=args.weights_dir,
            model_name=args.model,
        )
        print(f"\n✓ Model loaded successfully!")
    except ImportError as e:
        print(f"\n⚠️  Import error (expected if decoder.py not available): {e}")