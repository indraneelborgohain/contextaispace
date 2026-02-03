import gc
import json
import os
import torch
from pathlib import Path
from safetensors import safe_open
from typing import Dict, Tuple, List


def tensor_stats(t: torch.Tensor) -> Dict:
    """Get statistics for a tensor to verify correctness."""
    t_float = t.float()
    return {
        "shape": tuple(t.shape),
        "dtype": str(t.dtype),
        "min": t_float.min().item(),
        "max": t_float.max().item(),
        "mean": t_float.mean().item(),
        "std": t_float.std().item(),
        "norm": t_float.norm().item(),
        "nonzero": (t != 0).sum().item(),
    }


def print_tensor_stats(name: str, t: torch.Tensor):
    """Print tensor statistics in a readable format."""
    stats = tensor_stats(t)
    print(f"  {name}:")
    print(f"    shape={stats['shape']}, dtype={stats['dtype']}")
    print(f"    min={stats['min']:.6f}, max={stats['max']:.6f}, mean={stats['mean']:.6f}, std={stats['std']:.6f}")
    print(f"    norm={stats['norm']:.2f}, nonzero={stats['nonzero']}/{t.numel()}")


def dequantize_mxfp4(blocks: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """
    Dequantize MXFP4 weights. Each uint8 contains 2 packed 4-bit values.
    blocks: [num_experts, out_features, num_blocks, 16] uint8
    scales: [num_experts, out_features, num_blocks] uint8 (actually E4M3 floats stored as uint8)
    returns: [num_experts, out_features, num_blocks * 32] bfloat16
    
    MXFP4 format:
    - 4-bit values are unsigned (0-15) and need to be centered around 0
    - The scale is an E4M3 float stored in uint8 format
    """
    # Unpack each uint8 into two 4-bit values (high nibble and low nibble)
    high = (blocks >> 4).to(torch.float32)       # upper 4 bits (0-15)
    low = (blocks & 0x0F).to(torch.float32)      # lower 4 bits (0-15)

    # Interleave high and low: [E, O, B, 16] -> [E, O, B, 32]
    unpacked = torch.stack([high, low], dim=-1)    # [E, O, B, 16, 2]
    E, O, B, _, _ = unpacked.shape
    unpacked = unpacked.reshape(E, O, B, 32)       # [E, O, B, 32]

    # CENTER the values around 0: convert 0-15 to -8 to +7
    # This is crucial for proper weight distribution!
    unpacked = unpacked - 8.0

    # Decode E4M3 scale from uint8
    # E4M3 has 4 exponent bits (bias=7) and 3 mantissa bits
    scale_uint8 = scales.to(torch.int32)
    
    # Extract sign, exponent, and mantissa
    sign = ((scale_uint8 >> 7) & 0x1).to(torch.float32)
    exp = ((scale_uint8 >> 3) & 0x0F).to(torch.float32)  # 4 exponent bits
    mant = (scale_uint8 & 0x07).to(torch.float32)        # 3 mantissa bits
    
    # Reconstruct the E4M3 float value
    # For E4M3: value = (-1)^sign * 2^(exp-7) * (1 + mant/8)
    # Handle special case: exp=0 is subnormal
    normal_value = (1.0 - 2.0 * sign) * torch.pow(2.0, exp - 7.0) * (1.0 + mant / 8.0)
    subnormal_value = (1.0 - 2.0 * sign) * torch.pow(2.0, -6.0) * (mant / 8.0)
    
    # Use subnormal for exp=0, otherwise normal
    scales_float = torch.where(exp == 0, subnormal_value, normal_value)
    
    # Apply scales
    scales_expanded = scales_float.unsqueeze(-1)  # [E, O, B, 1]
    weights = unpacked * scales_expanded           # [E, O, B, 32]

    # Flatten: [E, O, B * 32] -> [E, O, 2880] (90 * 32 = 2880)
    weights = weights.reshape(E, O, B * 32)

    return weights.to(torch.bfloat16) 

def get_shard_info(model_path: str) -> Tuple[Dict, List[str]]:
    """Get index and shard files from model path."""
    model_path = Path(model_path)
    index_file = model_path / "model.safetensors.index.json"

    if not index_file.exists():
        raise FileNotFoundError(f"No index file at {index_file}")

    with open(index_file, 'r') as f:
        index = json.load(f)

    shard_files = sorted(set(index["weight_map"].values()))
    return index, shard_files


def load_keys_from_shards(model_path: str, keys: List[str], device: str = "cpu") -> Dict[str, torch.Tensor]:
    """Load only specific keys from sharded safetensors (memory efficient)."""
    model_path = Path(model_path)
    index, shard_files = get_shard_info(str(model_path))
    weight_map = index["weight_map"]
    
    # Group keys by shard
    keys_by_shard = {}
    for key in keys:
        if key in weight_map:
            shard = weight_map[key]
            if shard not in keys_by_shard:
                keys_by_shard[shard] = []
            keys_by_shard[shard].append(key)
    
    result = {}
    for shard, shard_keys in keys_by_shard.items():
        shard_path = model_path / shard
        with safe_open(str(shard_path), framework="pt", device=device) as f:
            for key in shard_keys:
                result[key] = f.get_tensor(key)
    
    return result


def load_sharded_state_dict(model_path: str, device: str = "cuda:0") -> Dict[str, torch.Tensor]:
    """Load from sharded safetensors using the index file."""
    model_path = Path(model_path)
    index_file = model_path / "model.safetensors.index.json"

    if not index_file.exists():
        raise FileNotFoundError(f"No index file at {index_file}")

    with open(index_file, 'r') as f:
        index = json.load(f)

    # Get unique shard filenames from the weight_map
    shard_files = sorted(set(index["weight_map"].values()))
    print(f"Found {len(shard_files)} shards: {shard_files}")

    state_dict = {}
    for shard in shard_files:
        shard_path = model_path / shard
        print(f"  Loading shard: {shard}")
        with safe_open(str(shard_path), framework="pt", device=device) as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

    return state_dict


def remap_layer_weights(
    model_path: str,
    layer_idx: int,
    device: str = "cpu"
) -> Dict[str, torch.Tensor]:
    """
    Load and remap weights for a single layer (memory efficient).
    Returns remapped state_dict entries for just this layer.
    """
    prefix = f"model.layers.{layer_idx}"
    new_prefix = f"layers.{layer_idx}"
    
    # Keys we need for this layer
    layer_keys = [
        f"{prefix}.input_layernorm.weight",
        f"{prefix}.post_attention_layernorm.weight",
        f"{prefix}.self_attn.q_proj.weight",
        f"{prefix}.self_attn.q_proj.bias",
        f"{prefix}.self_attn.k_proj.weight",
        f"{prefix}.self_attn.k_proj.bias",
        f"{prefix}.self_attn.v_proj.weight",
        f"{prefix}.self_attn.v_proj.bias",
        f"{prefix}.self_attn.o_proj.weight",
        f"{prefix}.self_attn.o_proj.bias",
        f"{prefix}.self_attn.sinks",
        f"{prefix}.mlp.router.weight",
        f"{prefix}.mlp.router.bias",
        f"{prefix}.mlp.experts.gate_up_proj_blocks",
        f"{prefix}.mlp.experts.gate_up_proj_scales",
        f"{prefix}.mlp.experts.gate_up_proj_bias",
        f"{prefix}.mlp.experts.down_proj_blocks",
        f"{prefix}.mlp.experts.down_proj_scales",
        f"{prefix}.mlp.experts.down_proj_bias",
    ]
    
    # Load only this layer's keys
    layer_data = load_keys_from_shards(model_path, layer_keys, device=device)
    
    result = {}
    
    # --- Attention sinks ---
    sink_key = f"{prefix}.self_attn.sinks"
    if sink_key in layer_data:
        result[f"{new_prefix}.attn.sink_logit"] = layer_data[sink_key].to(torch.float32)
    
    # --- Layernorms: bf16 → float32 ---
    result[f"{new_prefix}.norm1.weight"] = layer_data[f"{prefix}.input_layernorm.weight"].to(torch.float32)
    result[f"{new_prefix}.norm2.weight"] = layer_data[f"{prefix}.post_attention_layernorm.weight"].to(torch.float32)
    
    # --- Q, K, V ---
    result[f"{new_prefix}.attn.q.weight"] = layer_data[f"{prefix}.self_attn.q_proj.weight"]
    result[f"{new_prefix}.attn.k.weight"] = layer_data[f"{prefix}.self_attn.k_proj.weight"]
    result[f"{new_prefix}.attn.v.weight"] = layer_data[f"{prefix}.self_attn.v_proj.weight"]
    result[f"{new_prefix}.attn.q.bias"] = layer_data[f"{prefix}.self_attn.q_proj.bias"]
    result[f"{new_prefix}.attn.k.bias"] = layer_data[f"{prefix}.self_attn.k_proj.bias"]
    result[f"{new_prefix}.attn.v.bias"] = layer_data[f"{prefix}.self_attn.v_proj.bias"]
    
    # --- Output projection ---
    result[f"{new_prefix}.attn.o.weight"] = layer_data[f"{prefix}.self_attn.o_proj.weight"]
    result[f"{new_prefix}.attn.o.bias"] = layer_data[f"{prefix}.self_attn.o_proj.bias"]
    
    # --- Router / Gate ---
    result[f"{new_prefix}.moe.router.weight"] = layer_data[f"{prefix}.mlp.router.weight"]
    result[f"{new_prefix}.moe.router.bias"] = layer_data[f"{prefix}.mlp.router.bias"]
    
    # --- Dequantize experts ---
    gate_up_blocks = layer_data[f"{prefix}.mlp.experts.gate_up_proj_blocks"]
    gate_up_scales = layer_data[f"{prefix}.mlp.experts.gate_up_proj_scales"]
    gate_up_weight = dequantize_mxfp4(gate_up_blocks, gate_up_scales)
    gate_up_bias = layer_data[f"{prefix}.mlp.experts.gate_up_proj_bias"]
    
    # Free quantized tensors immediately
    del gate_up_blocks, gate_up_scales
    
    down_blocks = layer_data[f"{prefix}.mlp.experts.down_proj_blocks"]
    down_scales = layer_data[f"{prefix}.mlp.experts.down_proj_scales"]
    down_weight = dequantize_mxfp4(down_blocks, down_scales)
    down_bias = layer_data[f"{prefix}.mlp.experts.down_proj_bias"]
    
    # Free quantized tensors
    del down_blocks, down_scales
    del layer_data
    gc.collect()
    
    # Align to MoE param layout
    result[f"{new_prefix}.moe.W_in"] = gate_up_weight.transpose(1, 2)
    result[f"{new_prefix}.moe.b_in"] = gate_up_bias
    result[f"{new_prefix}.moe.W_out"] = down_weight.transpose(1, 2)
    result[f"{new_prefix}.moe.b_out"] = down_bias
    
    return result


def remap_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Remap checkpoint keys to match the GPT-OSS-20B Transformer naming,
    dequantize MXFP4 experts, and align tensor layouts.
    
    HuggingFace checkpoint keys -> gptoss20B.py model keys:
      model.layers.{i}.input_layernorm.weight         -> layers.{i}.norm1.weight
      model.layers.{i}.post_attention_layernorm.weight -> layers.{i}.norm2.weight
      model.layers.{i}.self_attn.q_proj.weight/bias   -> layers.{i}.attn.q.weight/bias
      model.layers.{i}.self_attn.k_proj.weight/bias   -> layers.{i}.attn.k.weight/bias
      model.layers.{i}.self_attn.v_proj.weight/bias   -> layers.{i}.attn.v.weight/bias
      model.layers.{i}.self_attn.o_proj.weight/bias   -> layers.{i}.attn.o.weight/bias
      model.layers.{i}.self_attn.sinks                -> layers.{i}.attn.sink_logit
      model.layers.{i}.mlp.router.weight/bias         -> layers.{i}.moe.router.weight/bias
      model.layers.{i}.mlp.experts.*                  -> layers.{i}.moe.W_in/b_in/W_out/b_out
      model.embed_tokens.weight                       -> embed.weight
      model.norm.weight                               -> norm_f.weight
      lm_head.weight                                  -> lm_head.weight
    """
    new_state_dict = {}

    # Find all layer indices
    layer_indices = set()
    for key in state_dict:
        if key.startswith("model.layers."):
            parts = key.split(".")
            layer_indices.add(int(parts[2]))

    for layer_idx in sorted(layer_indices):
        prefix = f"model.layers.{layer_idx}"
        new_prefix = f"layers.{layer_idx}"

        # --- Attention sinks (direct copy) ---
        if f"{prefix}.self_attn.sinks" in state_dict:
            new_state_dict[f"{new_prefix}.attn.sink_logit"] = (
                state_dict[f"{prefix}.self_attn.sinks"].to(torch.float32)
            )

        # --- Layernorms: bf16 → float32 ---
        new_state_dict[f"{new_prefix}.norm1.weight"] = (
            state_dict[f"{prefix}.input_layernorm.weight"].to(torch.float32)
        )
        new_state_dict[f"{new_prefix}.norm2.weight"] = (
            state_dict[f"{prefix}.post_attention_layernorm.weight"].to(torch.float32)
        )

        # --- Q, K, V (direct copy) ---
        new_state_dict[f"{new_prefix}.attn.q.weight"] = state_dict[f"{prefix}.self_attn.q_proj.weight"]
        new_state_dict[f"{new_prefix}.attn.k.weight"] = state_dict[f"{prefix}.self_attn.k_proj.weight"]
        new_state_dict[f"{new_prefix}.attn.v.weight"] = state_dict[f"{prefix}.self_attn.v_proj.weight"]
        new_state_dict[f"{new_prefix}.attn.q.bias"] = state_dict[f"{prefix}.self_attn.q_proj.bias"]
        new_state_dict[f"{new_prefix}.attn.k.bias"] = state_dict[f"{prefix}.self_attn.k_proj.bias"]
        new_state_dict[f"{new_prefix}.attn.v.bias"] = state_dict[f"{prefix}.self_attn.v_proj.bias"]

        # --- Output projection (direct copy) ---
        new_state_dict[f"{new_prefix}.attn.o.weight"] = state_dict[f"{prefix}.self_attn.o_proj.weight"]
        new_state_dict[f"{new_prefix}.attn.o.bias"] = state_dict[f"{prefix}.self_attn.o_proj.bias"]

        # --- Router / Gate (direct copy) ---
        new_state_dict[f"{new_prefix}.moe.router.weight"] = state_dict[f"{prefix}.mlp.router.weight"]
        new_state_dict[f"{new_prefix}.moe.router.bias"] = state_dict[f"{prefix}.mlp.router.bias"]

        # --- Dequantize and split experts ---
        # gate_up_proj: [E, 5760, 90, 16] → dequant → [E, 5760, 2880]
        gate_up_blocks = state_dict[f"{prefix}.mlp.experts.gate_up_proj_blocks"]
        gate_up_scales = state_dict[f"{prefix}.mlp.experts.gate_up_proj_scales"]
        gate_up_weight = dequantize_mxfp4(gate_up_blocks, gate_up_scales)  # [E, 5760, 2880]
        gate_up_bias = state_dict[f"{prefix}.mlp.experts.gate_up_proj_bias"]  # [E, 5760]

        # down_proj: [E, 2880, 90, 16] → dequant → [E, 2880, 2880]
        down_blocks = state_dict[f"{prefix}.mlp.experts.down_proj_blocks"]
        down_scales = state_dict[f"{prefix}.mlp.experts.down_proj_scales"]
        down_weight = dequantize_mxfp4(down_blocks, down_scales)  # [E, 2880, 2880]
        down_bias = state_dict[f"{prefix}.mlp.experts.down_proj_bias"]  # [E, 2880]

        # Align to MoE param layout
        # W_in: (E, H, 2*FF) where checkpoint is (E, 2*FF, H)
        new_state_dict[f"{new_prefix}.moe.W_in"] = gate_up_weight.transpose(1, 2)
        new_state_dict[f"{new_prefix}.moe.b_in"] = gate_up_bias
        # W_out: (E, FF, H) where checkpoint is (E, H, FF)
        new_state_dict[f"{new_prefix}.moe.W_out"] = down_weight.transpose(1, 2)
        new_state_dict[f"{new_prefix}.moe.b_out"] = down_bias

        print(f"  ✅ Layer {layer_idx} remapped")

    # --- Embedding (direct copy) ---
    if "model.embed_tokens.weight" in state_dict:
        new_state_dict["embed.weight"] = state_dict["model.embed_tokens.weight"]
    
    # --- Final layernorm / LM head (check what keys exist) ---
    if "model.norm.weight" in state_dict:
        new_state_dict["norm_f.weight"] = state_dict["model.norm.weight"].to(torch.float32)
    if "lm_head.weight" in state_dict:
        new_state_dict["lm_head.weight"] = state_dict["lm_head.weight"]

    return new_state_dict


def load_from_huggingface(
    model_path: str,
    config=None,
    device: str = "cuda:0",
    strict: bool = False,
    low_memory: bool = True,
):
    """
    Load GPT-OSS from sharded MXFP4 checkpoint into your custom Transformer.
    
    Args:
        model_path: Path to the model directory with safetensors files.
        config: Optional ModelConfig. If None, loads from config.json.
        device: Target device for the final model.
        strict: Whether to strictly enforce state_dict keys match.
        low_memory: If True, load and process one layer at a time to reduce peak memory.
    """
    from architecture.gptoss20B import Transformer, ModelConfig, RopeScalingConfig

    model_path = Path(model_path)
    if config is None:
        config_file = model_path / "config.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"No config.json found at {config_file}")
        
        # Handle HF config -> ModelConfig mapping
        # 1. swiglu_limit -> swiglu_clip
        if "swiglu_limit" in config_dict and "swiglu_clip" not in config_dict:
            config_dict["swiglu_clip"] = config_dict.pop("swiglu_limit")
        
        # 2. rope_scaling dict -> RopeScalingConfig
        if "rope_scaling" in config_dict and isinstance(config_dict["rope_scaling"], dict):
            rs = config_dict["rope_scaling"]
            # HF uses 'factor' key
            factor = rs.get("factor", 32.0)
            config_dict["rope_scaling"] = RopeScalingConfig(factor=factor)
        
        # Filter to only keys ModelConfig accepts
        import inspect
        valid_keys = inspect.signature(ModelConfig.__init__).parameters.keys()
        filtered_config = {k: v for k, v in config_dict.items() if k in valid_keys}
        print(f"Using config keys: {list(filtered_config.keys())}")
        print(f"Skipped keys: {[k for k in config_dict if k not in valid_keys]}")
        config = ModelConfig(**filtered_config)
    
    # Init model on CPU first with empty weights to save memory
    print("Initializing model structure...")
    model = Transformer(config)
    
    num_layers = config.num_hidden_layers
    
    if low_memory:
        # ==================== LOW MEMORY MODE ====================
        # Load and process one layer at a time, immediately loading into model
        print(f"Loading checkpoint in low-memory mode ({num_layers} layers)...")
        
        # Load embedding and final layers first (small)
        print("  Loading embedding and final layers...")
        embed_keys = ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]
        embed_data = load_keys_from_shards(str(model_path), embed_keys, device="cpu")
        
        if "model.embed_tokens.weight" in embed_data:
            model.embed.weight.data.copy_(embed_data["model.embed_tokens.weight"])
        if "model.norm.weight" in embed_data:
            model.norm_f.weight.data.copy_(embed_data["model.norm.weight"].to(torch.float32))
        if "lm_head.weight" in embed_data:
            model.lm_head.weight.data.copy_(embed_data["lm_head.weight"])
        
        del embed_data
        gc.collect()
        print("  ✅ Embedding layers loaded")
        
        # Process each transformer layer one at a time
        print("Remapping and loading transformer layers...")
        for layer_idx in range(num_layers):
            # Load, remap, and dequantize this layer
            layer_state_dict = remap_layer_weights(str(model_path), layer_idx, device="cpu")
            
            # Copy weights directly into model parameters
            layer = model.layers[layer_idx]
            
            layer.norm1.weight.data.copy_(layer_state_dict[f"layers.{layer_idx}.norm1.weight"])
            layer.norm2.weight.data.copy_(layer_state_dict[f"layers.{layer_idx}.norm2.weight"])
            
            layer.attn.q.weight.data.copy_(layer_state_dict[f"layers.{layer_idx}.attn.q.weight"])
            layer.attn.k.weight.data.copy_(layer_state_dict[f"layers.{layer_idx}.attn.k.weight"])
            layer.attn.v.weight.data.copy_(layer_state_dict[f"layers.{layer_idx}.attn.v.weight"])
            layer.attn.q.bias.data.copy_(layer_state_dict[f"layers.{layer_idx}.attn.q.bias"])
            layer.attn.k.bias.data.copy_(layer_state_dict[f"layers.{layer_idx}.attn.k.bias"])
            layer.attn.v.bias.data.copy_(layer_state_dict[f"layers.{layer_idx}.attn.v.bias"])
            layer.attn.o.weight.data.copy_(layer_state_dict[f"layers.{layer_idx}.attn.o.weight"])
            layer.attn.o.bias.data.copy_(layer_state_dict[f"layers.{layer_idx}.attn.o.bias"])
            
            sink_key = f"layers.{layer_idx}.attn.sink_logit"
            if sink_key in layer_state_dict:
                layer.attn.sink_logit.data.copy_(layer_state_dict[sink_key])
            
            layer.moe.router.weight.data.copy_(layer_state_dict[f"layers.{layer_idx}.moe.router.weight"])
            layer.moe.router.bias.data.copy_(layer_state_dict[f"layers.{layer_idx}.moe.router.bias"])
            layer.moe.W_in.data.copy_(layer_state_dict[f"layers.{layer_idx}.moe.W_in"])
            layer.moe.b_in.data.copy_(layer_state_dict[f"layers.{layer_idx}.moe.b_in"])
            layer.moe.W_out.data.copy_(layer_state_dict[f"layers.{layer_idx}.moe.W_out"])
            layer.moe.b_out.data.copy_(layer_state_dict[f"layers.{layer_idx}.moe.b_out"])
            
            # Free memory immediately
            del layer_state_dict
            gc.collect()
            
            print(f"  ✅ Layer {layer_idx} loaded")
        
        print(f"\n✅ Successfully loaded GPT-OSS model (low-memory mode)")
    
    else:
        # ==================== ORIGINAL MODE (high memory) ====================
        # Load sharded checkpoint
        print("Loading sharded checkpoint...")
        raw_state_dict = load_sharded_state_dict(str(model_path), device="cpu")

        # Remap keys + dequantize
        print("Remapping keys and dequantizing experts...")
        state_dict = remap_state_dict(raw_state_dict)
        
        # Free original state dict
        del raw_state_dict
        gc.collect()

        # Debug: check for mismatches before loading
        model_keys = set(model.state_dict().keys())
        checkpoint_keys = set(state_dict.keys())
        missing = model_keys - checkpoint_keys
        unexpected = checkpoint_keys - model_keys

        print(f"\n{'='*60}")
        print(f"Weight Loading Debug Info")
        print(f"{'='*60}")
        print(f"Model has {len(model_keys)} parameters")
        print(f"Checkpoint has {len(checkpoint_keys)} parameters")
        print(f"Matching keys: {len(model_keys & checkpoint_keys)}")
        
        if missing:
            print(f"\n⚠️  Missing keys ({len(missing)}) - in model but NOT in checkpoint:")
            for k in sorted(missing):
                model_shape = model.state_dict()[k].shape
                print(f"    {k}: {model_shape}")
        else:
            print(f"\n✅ No missing keys")
            
        if unexpected:
            print(f"\n⚠️  Unexpected keys ({len(unexpected)}) - in checkpoint but NOT in model:")
            for k in sorted(unexpected):
                ckpt_shape = state_dict[k].shape
                print(f"    {k}: {ckpt_shape}")
        else:
            print(f"\n✅ No unexpected keys")
        
        # Check for shape mismatches in matching keys
        shape_mismatches = []
        for k in model_keys & checkpoint_keys:
            model_shape = model.state_dict()[k].shape
            ckpt_shape = state_dict[k].shape
            if model_shape != ckpt_shape:
                shape_mismatches.append((k, model_shape, ckpt_shape))
        
        if shape_mismatches:
            print(f"\n❌ Shape mismatches ({len(shape_mismatches)}):")
            for k, model_shape, ckpt_shape in shape_mismatches:
                print(f"    {k}: model={model_shape} vs checkpoint={ckpt_shape}")
        else:
            print(f"\n✅ All matching keys have correct shapes")
        
        print(f"{'='*60}\n")

        # Load
        model.load_state_dict(state_dict, strict=strict)
        
        # Free state dict
        del state_dict
        gc.collect()
        
        print(f"\n✅ Successfully loaded GPT-OSS model")
    
    # Move to target device
    model = model.to(device)
    print(f"Model moved to {device}")
    
    return model


def verify_loaded_weights(model, model_path: str, layer_to_check: int = 0):
    """
    Verify that weights were loaded correctly by comparing model weights
    against the original checkpoint. Checks a single layer to save memory.
    
    Args:
        model: The loaded Transformer model
        model_path: Path to the checkpoint directory
        layer_to_check: Which layer index to verify (default: 0)
    """
    print(f"\n{'='*60}")
    print(f"Weight Verification for Layer {layer_to_check}")
    print(f"{'='*60}")
    
    model_path = Path(model_path)
    prefix = f"model.layers.{layer_to_check}"
    
    # Load original checkpoint keys for this layer
    check_keys = [
        f"{prefix}.input_layernorm.weight",
        f"{prefix}.self_attn.q_proj.weight",
        f"{prefix}.self_attn.q_proj.bias",
        f"{prefix}.self_attn.k_proj.weight",
        f"{prefix}.mlp.router.weight",
        f"{prefix}.mlp.experts.gate_up_proj_blocks",
        f"{prefix}.mlp.experts.gate_up_proj_scales",
        f"{prefix}.mlp.experts.gate_up_proj_bias",
    ]
    
    ckpt_data = load_keys_from_shards(str(model_path), check_keys, device="cpu")
    
    layer = model.layers[layer_to_check]
    
    # 1. Check layernorm (should be identical after dtype conversion)
    print("\n1. LayerNorm weight:")
    ckpt_ln = ckpt_data[f"{prefix}.input_layernorm.weight"].to(torch.float32)
    model_ln = layer.norm1.weight.data.cpu()
    print_tensor_stats("  Checkpoint", ckpt_ln)
    print_tensor_stats("  Model", model_ln)
    ln_match = torch.allclose(ckpt_ln, model_ln, atol=1e-6)
    print(f"  ✅ Match: {ln_match}" if ln_match else f"  ❌ MISMATCH!")
    
    # 2. Check Q projection weight (should be identical)
    print("\n2. Q projection weight:")
    ckpt_q = ckpt_data[f"{prefix}.self_attn.q_proj.weight"]
    model_q = layer.attn.q.weight.data.cpu()
    print_tensor_stats("  Checkpoint", ckpt_q)
    print_tensor_stats("  Model", model_q)
    q_match = torch.allclose(ckpt_q.float(), model_q.float(), atol=1e-5)
    print(f"  ✅ Match: {q_match}" if q_match else f"  ❌ MISMATCH!")
    
    # 3. Check Q projection bias
    print("\n3. Q projection bias:")
    ckpt_qb = ckpt_data[f"{prefix}.self_attn.q_proj.bias"]
    model_qb = layer.attn.q.bias.data.cpu()
    print_tensor_stats("  Checkpoint", ckpt_qb)
    print_tensor_stats("  Model", model_qb)
    qb_match = torch.allclose(ckpt_qb.float(), model_qb.float(), atol=1e-5)
    print(f"  ✅ Match: {qb_match}" if qb_match else f"  ❌ MISMATCH!")
    
    # 4. Check router weight
    print("\n4. MoE Router weight:")
    ckpt_router = ckpt_data[f"{prefix}.mlp.router.weight"]
    model_router = layer.moe.router.weight.data.cpu()
    print_tensor_stats("  Checkpoint", ckpt_router)
    print_tensor_stats("  Model", model_router)
    router_match = torch.allclose(ckpt_router.float(), model_router.float(), atol=1e-5)
    print(f"  ✅ Match: {router_match}" if router_match else f"  ❌ MISMATCH!")
    
    # 5. Check dequantized expert weights
    print("\n5. Expert gate_up_proj (dequantized):")
    gate_up_blocks = ckpt_data[f"{prefix}.mlp.experts.gate_up_proj_blocks"]
    gate_up_scales = ckpt_data[f"{prefix}.mlp.experts.gate_up_proj_scales"]
    ckpt_gate_up = dequantize_mxfp4(gate_up_blocks, gate_up_scales)
    # W_in in model is transposed: (E, H, 2*FF) vs checkpoint (E, 2*FF, H)
    model_W_in = layer.moe.W_in.data.cpu()
    print(f"  Checkpoint dequantized shape: {ckpt_gate_up.shape}")
    print(f"  Model W_in shape: {model_W_in.shape}")
    # Transpose model back to compare
    model_W_in_t = model_W_in.transpose(1, 2)
    print_tensor_stats("  Checkpoint (gate_up)", ckpt_gate_up)
    print_tensor_stats("  Model W_in (transposed back)", model_W_in_t)
    expert_match = torch.allclose(ckpt_gate_up.float(), model_W_in_t.float(), atol=1e-4)
    print(f"  ✅ Match: {expert_match}" if expert_match else f"  ❌ MISMATCH!")
    
    # 6. Check expert bias
    print("\n6. Expert gate_up_proj bias:")
    ckpt_bias = ckpt_data[f"{prefix}.mlp.experts.gate_up_proj_bias"]
    model_bias = layer.moe.b_in.data.cpu()
    print_tensor_stats("  Checkpoint", ckpt_bias)
    print_tensor_stats("  Model b_in", model_bias)
    bias_match = torch.allclose(ckpt_bias.float(), model_bias.float(), atol=1e-5)
    print(f"  ✅ Match: {bias_match}" if bias_match else f"  ❌ MISMATCH!")
    
    # 7. Check embedding
    print("\n7. Embedding weight:")
    embed_data = load_keys_from_shards(str(model_path), ["model.embed_tokens.weight"], device="cpu")
    ckpt_embed = embed_data["model.embed_tokens.weight"]
    model_embed = model.embed.weight.data.cpu()
    print_tensor_stats("  Checkpoint", ckpt_embed)
    print_tensor_stats("  Model", model_embed)
    embed_match = torch.allclose(ckpt_embed.float(), model_embed.float(), atol=1e-5)
    print(f"  ✅ Match: {embed_match}" if embed_match else f"  ❌ MISMATCH!")
    
    print(f"\n{'='*60}")
    all_match = all([ln_match, q_match, qb_match, router_match, expert_match, bias_match, embed_match])
    if all_match:
        print("✅ All weights verified successfully!")
    else:
        print("❌ Some weights don't match - check the details above")
    print(f"{'='*60}\n")
    
    return all_match