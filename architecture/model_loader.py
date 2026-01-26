"""
Model loader utility for loading pretrained GPT models.
Supports loading from local checkpoints with configurable paths.
Supports both PyTorch (.pt) and SafeTensors (.safetensors) formats.
"""

import os
import torch
from pathlib import Path
from typing import Optional, Union
from architecture.gptoss import Transformer, ModelConfig
from architecture.gpt2 import GPTModel, GPT_CONFIG

try:
    from safetensors.torch import load_file as load_safetensors
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("⚠️  safetensors not available. Install with: pip install safetensors")


def load_state_dict_from_file(file_path: str, device: str = "cuda:0") -> dict:
    """
    Load state dict from either .pt or .safetensors format.
    
    Args:
        file_path: Path to checkpoint file
        device: Device to load tensors on
        
    Returns:
        State dictionary
    """
    if file_path.endswith('.safetensors'):
        if not SAFETENSORS_AVAILABLE:
            raise ImportError(
                "safetensors is required to load .safetensors files. "
                "Install with: pip install safetensors"
            )
        print(f"📦 Loading from SafeTensors format")
        state_dict = load_safetensors(file_path, device=device)
    else:
        print(f"📦 Loading from PyTorch format")
        state_dict = torch.load(file_path, map_location=device)
    
    return state_dict


def load_pretrained_gptoss(
    checkpoint_path: str = "models/gptoss_best.pt",
    config: ModelConfig = None,
    device: str = "cuda:0",
    strict: bool = False
) -> Transformer:
    """
    Load a pretrained GPT-OSS model from checkpoint.
    
    Args:
        checkpoint_path: Path to the .pt checkpoint file
        config: ModelConfig object. If None, uses default config
        device: Device to load model on
        strict: Whether to strictly enforce state_dict keys match
        
    Returns:
        Loaded Transformer model
    """
    if config is None:
        # Default configuration matching training setup
        config = ModelConfig(
            num_attention_heads=8,
            num_key_value_heads=4,
            num_experts=4,
            experts_per_token=1,
            num_hidden_layers=12,
            hidden_size=1024,
            intermediate_size=1024
        )
    
    # Initialize model
    model = Transformer(config, device=device)
    
    # Load checkpoint if it exists
    if os.path.exists(checkpoint_path):
        print(f"Loading pretrained model from: {checkpoint_path}")
        state_dict = load_state_dict_from_file(checkpoint_path, device=device)
        model.load_state_dict(state_dict, strict=strict)
        print(f"✅ Successfully loaded pretrained model")
    else:
        print(f"⚠️  Checkpoint not found at {checkpoint_path}")
        print(f"   Initializing model with random weights")
    
    return model


def load_pretrained_gpt2(
    checkpoint_path: str = "models/gpt2_best.pt",
    config: dict = None,
    device: str = "cuda:0",
    strict: bool = False
) -> GPTModel:
    """
    Load a pretrained GPT2 model from checkpoint.
    
    Args:
        checkpoint_path: Path to the .pt checkpoint file
        config: Config dict. If None, uses default GPT_CONFIG
        device: Device to load model on
        strict: Whether to strictly enforce state_dict keys match
        
    Returns:
        Loaded GPTModel
    """
    if config is None:
        config = GPT_CONFIG
    
    # Initialize model
    model = GPTModel(config).to(device)
    
    # Load checkpoint if it exists
    if os.path.exists(checkpoint_path):
        print(f"Loading pretrained GPT2 model from: {checkpoint_path}")
        state_dict = load_state_dict_from_file(checkpoint_path, device=device)
        model.load_state_dict(state_dict, strict=strict)
        print(f"✅ Successfully loaded pretrained GPT2 model")
    else:
        print(f"⚠️  Checkpoint not found at {checkpoint_path}")
        print(f"   Initializing model with random weights")
    
    return model


def save_model(
    model, 
    save_path: str, 
    create_dir: bool = True,
    use_safetensors: bool = False
):
    """
    Save model checkpoint in .pt or .safetensors format.
    
    Args:
        model: PyTorch model to save
        save_path: Path where to save the checkpoint
        create_dir: Whether to create directory if it doesn't exist
        use_safetensors: Whether to save in safetensors format
    """
    if create_dir:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    state_dict = model.state_dict()
    
    if use_safetensors:
        if not SAFETENSORS_AVAILABLE:
            print("⚠️  safetensors not available. Saving as .pt instead")
            torch.save(state_dict, save_path)
        else:
            from safetensors.torch import save_file as save_safetensors
            # Ensure path has .safetensors extension
            if not save_path.endswith('.safetensors'):
                save_path = save_path.replace('.pt', '.safetensors')
            save_safetensors(state_dict, save_path)
            print(f"💾 Model saved to SafeTensors: {save_path}")
    else:
        torch.save(state_dict, save_path)
        print(f"💾 Model saved to: {save_path}")


def list_available_checkpoints(models_dir: str = "models") -> list:
    """
    List all available model checkpoints in the models directory.
    Supports both .pt and .safetensors formats.
    
    Args:
        models_dir: Directory to search for checkpoints
        
    Returns:
        List of checkpoint file paths
    """
    if not os.path.exists(models_dir):
        print(f"Models directory not found: {models_dir}")
        return []
    
    # Search for both .pt and .safetensors files
    pt_checkpoints = list(Path(models_dir).glob("*.pt"))
    safetensors_checkpoints = list(Path(models_dir).glob("*.safetensors"))
    checkpoints = pt_checkpoints + safetensors_checkpoints
    
    if checkpoints:
        print(f"\n📦 Available checkpoints in {models_dir}:")
        for ckpt in sorted(checkpoints):
            size_mb = ckpt.stat().st_size / (1024 * 1024)
            file_type = "SafeTensors" if ckpt.suffix == ".safetensors" else "PyTorch"
            print(f"   - {ckpt.name} ({size_mb:.2f} MB) [{file_type}]")
    else:
        print(f"No checkpoints found in {models_dir}")
    
    return [str(ckpt) for ckpt in checkpoints]




def load_from_huggingface(
    model_path: str,
    config: ModelConfig = None,
    device: str = "cuda:0",
    strict: bool = False,
    safetensors_filename: str = "model.safetensors"
) -> Transformer:
    """
    Load a model from a HuggingFace downloaded directory.
    
    Args:
        model_path: Path to directory containing model files
        config: ModelConfig object. If None, will try to load from config.json
        device: Device to load model on
        strict: Whether to strictly enforce state_dict keys match
        safetensors_filename: Name of the safetensors file (default: "model.safetensors")
        
    Returns:
        Loaded Transformer model
        
    Example:
        # After downloading from HuggingFace
        model = load_from_huggingface(
            model_path="models/downloaded_model",
            config=config,
            device="cuda:0"
        )
    """
    import json
    
    model_path = Path(model_path)
    
    # Try to load config from config.json if not provided
    if config is None:
        config_file = model_path / "config.json"
        if config_file.exists():
            print(f"Loading config from: {config_file}")
            with open(config_file, 'r') as f:
                config_dict = json.load(f)
            # Try to map HuggingFace config to ModelConfig
            config = ModelConfig(**config_dict)
        else:
            raise ValueError(
                "No config provided and config.json not found. "
                "Please provide a ModelConfig object."
            )
    
    # Initialize model
    model = Transformer(config, device=device)
    
    # Look for safetensors or pytorch_model.bin
    safetensors_path = model_path / safetensors_filename
    pytorch_path = model_path / "pytorch_model.bin"
    
    if safetensors_path.exists():
        print(f"Loading from HuggingFace SafeTensors: {safetensors_path}")
        state_dict = load_state_dict_from_file(str(safetensors_path), device=device)
    elif pytorch_path.exists():
        print(f"Loading from HuggingFace PyTorch: {pytorch_path}")
        state_dict = load_state_dict_from_file(str(pytorch_path), device=device)
    else:
        raise FileNotFoundError(
            f"No model file found in {model_path}. "
            f"Expected {safetensors_filename} or pytorch_model.bin"
        )


    
    model.load_state_dict(state_dict, strict=strict)
    print(f"✅ Successfully loaded model from HuggingFace directory")
    
    return model
def get_model_info(model) -> dict:
    """
    Get information about a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info = {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
    }
    
    print(f"\n🔍 Model Information:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Estimated size: {info['model_size_mb']:.2f} MB")
    
    return info
