import torch
from torch.nn import functional as F
import os
import argparse

from architecture.tokenizer import get_tokenizer
from architecture.gptoss import Transformer, ModelConfig




context_len=8192
tokenizer= get_tokenizer()

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    return tokenizer.decode(token_ids.tolist())



def generate_text(model, prompt, max_tokens=100, temperature=0.8, top_k=50):
    """Generate text from a prompt using trained model."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    # Tokenize input
    
    idx = text_to_token_ids(prompt,tokenizer).to(device)
    # Generate
    for _ in range(max_tokens):
        idx_cond = idx[-context_len:]
        with torch.inference_mode():
            logits= model(idx_cond)
        logits = logits[-1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[[-1]]] = -float('Inf')

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=0)

    
    
       
    
    # Decode and return
    result = token_ids_to_text(idx,tokenizer)
    return result


def load_model_and_generate(
    checkpoint_path,
    prompt="Once upon a time",
    max_tokens=200,
    temperature=0.8,
    top_k=200,
    device=None
):
    """
    Load a trained model from checkpoint and generate text.
    
    Args:
        checkpoint_path: Path to the checkpoint file (e.g., 'model/gptoss_best.pt')
        prompt: Text prompt to start generation
        max_tokens: Number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling parameter
        device: Device to use ('cuda:0', 'cpu', etc.). Auto-detect if None.
    
    Returns:
        Generated text string
    """
    # Auto-detect device
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model from: {checkpoint_path}")
    print(f"Using device: {device}")
    
    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract config
    if isinstance(checkpoint, dict) and "config" in checkpoint:
        config_dict = checkpoint["config"]
        # Build ModelConfig from saved args
        vocab_size = 201088  # o200k_harmony
        
        # Determine model size from config
        if "model_size" in config_dict:
            model_size = config_dict["model_size"]
        else:
            # Try to infer from saved model architecture
            model_size = "toy"  # default
        
        # Build config based on model size
        from train import build_config
        cfg = build_config(model_size, vocab_size)
        
    else:
        # Default config if not saved
        print("Warning: Config not found in checkpoint, using default toy config")
        cfg = ModelConfig(
            vocab_size=201088,
            hidden_size=512,
            num_hidden_layers=6,
            head_dim=64,
            num_attention_heads=8,
            num_key_value_heads=2,
            num_experts=4,
            experts_per_token=2,
            intermediate_size=512,
            sliding_window=64,
            initial_context_length=2048,
        )
    
    # Create model
    print(f"Creating model with config: {cfg.hidden_size}d, {cfg.num_hidden_layers} layers")
    model = Transformer(cfg, device=device)
    
    # Load weights
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if "iter_num" in checkpoint:
            print(f"Loaded checkpoint from iteration {checkpoint['iter_num']}")
        if "best_val_loss" in checkpoint:
            print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
    else:
        model.load_state_dict(checkpoint)
    
    print(f"Model loaded successfully!")
    print(f"\nGenerating text with prompt: '{prompt}'")
    print("=" * 70)
    
    # Generate text
    generated = generate_text(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k
    )
    
    return generated


def main():
    parser = argparse.ArgumentParser(description="Load model and generate text")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="model/gptoss_best.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time",
        help="Text prompt for generation"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=200,
        help="Number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=200,
        help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda:0, cpu, etc.)"
    )
    
    args = parser.parse_args()
    
    # Generate text
    generated_text = load_model_and_generate(
        checkpoint_path=args.checkpoint,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device
    )
    
    print(generated_text)
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()


# ========================================================================================
# GPT-OSS 20B Model Inference
# ========================================================================================

def load_gptoss20b_and_generate(
    weights_dir="architecture/open-gpt-oss/weights",
    prompt="Once upon a time",
    max_tokens=200,
    temperature=0.8,
    top_k=200,
    device=None
):
    """
    Load the GPT-OSS 20B model with downloaded weights and generate text.
    
    This function loads the official GPT-OSS 20B architecture and weights.
    Make sure you've downloaded the weights first using:
        python architecture/open-gpt-oss/download_weights.py
    
    Args:
        weights_dir: Directory containing the downloaded weights (config.json, model.safetensors, etc.)
        prompt: Text prompt to start generation
        max_tokens: Number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling parameter
        device: Device to use ('cuda:0', 'cpu', etc.). Auto-detect if None.
    
    Returns:
        Generated text string
    """
    import json
    from pathlib import Path
    
    # Import GPT-OSS model
    import sys
    sys.path.insert(0, 'architecture/open-gpt-oss')
    from model import Transformer, gpt_oss_20b_config
    
    # Auto-detect device
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading GPT-OSS 20B model from: {weights_dir}")
    print(f"Using device: {device}")
    
    weights_path = Path(weights_dir)
    
    # Check if weights directory exists
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights directory not found: {weights_dir}\n"
            "Please download weights first using:\n"
            "  python architecture/open-gpt-oss/download_weights.py"
        )
    
    # Load config
    config_file = weights_path / "config.json"
    if config_file.exists():
        print("Loading config from config.json...")
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        # Use the config from file if available
        print(f"Config loaded: {config_dict.get('model_type', 'gpt-oss')}")
    
    # Create model with 20B config
    print("Creating GPT-OSS 20B model...")
    cfg = gpt_oss_20b_config()
    model = Transformer(cfg)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params/1e9:.2f}B")
    
    # Load weights
    weights_file = None
    
    # Try to find safetensors or pytorch_model.bin
    for ext in ["*.safetensors", "*.bin", "*.pt"]:
        weight_files = list(weights_path.glob(ext))
        if weight_files:
            weights_file = weight_files[0]
            break
    
    if weights_file is None:
        raise FileNotFoundError(
            f"No weight files found in {weights_dir}\n"
            "Expected files: model.safetensors, pytorch_model.bin, or .pt files"
        )
    
    print(f"Loading weights from: {weights_file.name}")
    
    # Load weights based on file type
    if str(weights_file).endswith('.safetensors'):
        try:
            from safetensors.torch import load_file
            state_dict = load_file(str(weights_file))
            print("Loaded weights from safetensors")
        except ImportError:
            raise ImportError(
                "safetensors package required to load .safetensors files.\n"
                "Install with: pip install safetensors"
            )
    else:
        # Load .bin or .pt files
        state_dict = torch.load(weights_file, map_location=device)
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        print(f"Loaded weights from {weights_file.suffix}")
    
    # Load state dict into model
    try:
        model.load_state_dict(state_dict, strict=False)
        print("✅ Weights loaded successfully!")
    except Exception as e:
        print(f"⚠️ Warning loading weights (may still work): {e}")
    
    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    print(f"\nGenerating text with prompt: '{prompt}'")
    print("=" * 70)
    
    # Tokenizer
    tokenizer = get_tokenizer()
    
    # Tokenize prompt
    if prompt:
        tokens = tokenizer.encode(prompt)
    else:
        tokens = tokenizer.encode("\n")
    
    tokens = torch.tensor(tokens, device=device, dtype=torch.long).unsqueeze(0)  # Add batch dim
    
    # Generate
    with torch.no_grad():
        for _ in range(max_tokens):
            # Forward pass
            logits, _ = model(tokens, labels=None)
            next_logits = logits[0, -1, :]  # Get last token logits
            
            # Apply temperature
            if temperature != 1.0:
                next_logits = next_logits / max(1e-6, temperature)
            
            # Apply top-k filtering
            if top_k and top_k > 0:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[-1]] = -float("inf")
            
            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).unsqueeze(0)
            
            # Append to sequence
            tokens = torch.cat([tokens, next_token], dim=1)
            
            # Check for EOS if defined
            if cfg.eos_token_id is not None and next_token.item() == cfg.eos_token_id:
                break
    
    # Decode
    generated_text = tokenizer.decode(tokens[0].tolist())
    
    return generated_text


def main_gptoss20b():
    """Command line interface for GPT-OSS 20B inference"""
    parser = argparse.ArgumentParser(description="GPT-OSS 20B Text Generation")
    parser.add_argument(
        "--weights_dir",
        type=str,
        default="architecture/open-gpt-oss/weights",
        help="Directory containing downloaded GPT-OSS 20B weights"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time",
        help="Text prompt for generation"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=200,
        help="Number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=200,
        help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda:0, cpu, etc.)"
    )
    
    args = parser.parse_args()
    
    # Generate text
    generated_text = load_gptoss20b_and_generate(
        weights_dir=args.weights_dir,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device
    )
    
    print(generated_text)
    print("\n" + "=" * 70)


# Run GPT-OSS 20B inference if called with --gptoss20b flag
if __name__ == "__main__":
    import sys
    if "--gptoss20b" in sys.argv:
        sys.argv.remove("--gptoss20b")
        main_gptoss20b()
    else:
        main()
