import json
import torch
from pathlib import Path
from safetensors import safe_open
from typing import Dict, Tuple


def dequantize_mxfp4(blocks: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """
    Dequantize MXFP4 weights. Each uint8 contains 2 packed 4-bit values.
    blocks: [num_experts, out_features, num_blocks, 16] uint8
    scales: [num_experts, out_features, num_blocks] uint8
    returns: [num_experts, out_features, num_blocks * 32] bfloat16
    """
    # Unpack each uint8 into two 4-bit values (high nibble and low nibble)
    high = (blocks >> 4).to(torch.bfloat16)       # upper 4 bits
    low = (blocks & 0x0F).to(torch.bfloat16)      # lower 4 bits

    # Interleave high and low: [E, O, B, 16] -> [E, O, B, 32]
    unpacked = torch.stack([high, low], dim=-1)    # [E, O, B, 16, 2]
    E, O, B, _, _ = unpacked.shape
    unpacked = unpacked.reshape(E, O, B, 32)       # [E, O, B, 32]

    # Apply scales
    scales_bf16 = scales.to(torch.bfloat16).unsqueeze(-1)  # [E, O, B, 1]
    weights = unpacked * scales_bf16                        # [E, O, B, 32]

    # Flatten: [E, O, B * 32] -> [E, O, 2880] (90 * 32 = 2880) ✅
    weights = weights.reshape(E, O, B * 32)

    return weights 

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
):
    """Load GPT-OSS from sharded MXFP4 checkpoint into your custom Transformer."""
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
    # Init model on CPU first, then move to device after loading weights
    model = Transformer(config)
    # Load sharded checkpoint
    print("Loading sharded checkpoint...")
    raw_state_dict = load_sharded_state_dict(str(model_path), device="cpu")  # load to CPU first

    # Remap keys + dequantize
    print("Remapping keys and dequantizing experts...")
    state_dict = remap_state_dict(raw_state_dict)

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
    print(f"\n✅ Successfully loaded GPT-OSS model")
    
    # Move to target device
    model = model.to(device)
    print(f"Model moved to {device}")
    
    return model