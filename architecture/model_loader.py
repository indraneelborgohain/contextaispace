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

    shard_files = sorted(set(index["weight_map"].values()))

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
    Remap checkpoint keys to match your Transformer model's naming,
    fuse Q/K/V, dequantize MXFP4 experts, and split batched experts.
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
        new_prefix = f"block.{layer_idx}"

        # --- Attention sinks (direct copy) ---
        new_state_dict[f"{new_prefix}.attn.sinks"] = state_dict[f"{prefix}.self_attn.sinks"]

        # --- Layernorms: bf16 → float32 ---
        new_state_dict[f"{new_prefix}.attn.norm.scale"] = (
            state_dict[f"{prefix}.input_layernorm.weight"].to(torch.float32)
        )
        new_state_dict[f"{new_prefix}.mlp.norm.scale"] = (
            state_dict[f"{prefix}.post_attention_layernorm.weight"].to(torch.float32)
        )

        # --- Fuse Q, K, V into single qkv weight and bias ---
        q_weight = state_dict[f"{prefix}.self_attn.q_proj.weight"]  # [4096, 2880]
        k_weight = state_dict[f"{prefix}.self_attn.k_proj.weight"]  # [512, 2880]
        v_weight = state_dict[f"{prefix}.self_attn.v_proj.weight"]  # [512, 2880]
        new_state_dict[f"{new_prefix}.attn.qkv.weight"] = torch.cat([q_weight, k_weight, v_weight], dim=0)  # [5120, 2880]

        q_bias = state_dict[f"{prefix}.self_attn.q_proj.bias"]  # [4096]
        k_bias = state_dict[f"{prefix}.self_attn.k_proj.bias"]  # [512]
        v_bias = state_dict[f"{prefix}.self_attn.v_proj.bias"]  # [512]
        new_state_dict[f"{new_prefix}.attn.qkv.bias"] = torch.cat([q_bias, k_bias, v_bias], dim=0)  # [5120]

        # --- Output projection (direct copy) ---
        new_state_dict[f"{new_prefix}.attn.out.weight"] = state_dict[f"{prefix}.self_attn.o_proj.weight"]
        new_state_dict[f"{new_prefix}.attn.out.bias"] = state_dict[f"{prefix}.self_attn.o_proj.bias"]

        # --- Router / Gate (direct copy) ---
        new_state_dict[f"{new_prefix}.mlp.gate.weight"] = state_dict[f"{prefix}.mlp.router.weight"]
        new_state_dict[f"{new_prefix}.mlp.gate.bias"] = state_dict[f"{prefix}.mlp.router.bias"]

        # --- Dequantize and split experts ---
        # gate_up_proj: [32, 5760, 90, 16] → dequant → [32, 5760, 1440]
        gate_up_blocks = state_dict[f"{prefix}.mlp.experts.gate_up_proj_blocks"]
        gate_up_scales = state_dict[f"{prefix}.mlp.experts.gate_up_proj_scales"]
        gate_up_weight = dequantize_mxfp4(gate_up_blocks, gate_up_scales)  # [32, 5760, 1440]
        gate_up_bias = state_dict[f"{prefix}.mlp.experts.gate_up_proj_bias"]  # [32, 5760]

        # down_proj: [32, 2880, 90, 16] → dequant → [32, 2880, 1440]
        down_blocks = state_dict[f"{prefix}.mlp.experts.down_proj_blocks"]
        down_scales = state_dict[f"{prefix}.mlp.experts.down_proj_scales"]
        down_weight = dequantize_mxfp4(down_blocks, down_scales)  # [32, 2880, 1440]
        down_bias = state_dict[f"{prefix}.mlp.experts.down_proj_bias"]  # [32, 2880]

        # Split into individual experts
        num_experts = gate_up_weight.shape[0]  # 32
        for expert_idx in range(num_experts):
            # Expert .0 = gate_up_proj (up projection)
            new_state_dict[f"{new_prefix}.mlp.experts.{expert_idx}.0.weight"] = gate_up_weight[expert_idx]  # [5760, 1440]
            new_state_dict[f"{new_prefix}.mlp.experts.{expert_idx}.0.bias"] = gate_up_bias[expert_idx]      # [5760]

            # Expert .1 = down_proj (down projection)
            new_state_dict[f"{new_prefix}.mlp.experts.{expert_idx}.1.weight"] = down_weight[expert_idx]    # [2880, 1440]
            new_state_dict[f"{new_prefix}.mlp.experts.{expert_idx}.1.bias"] = down_bias[expert_idx]        # [2880]

        print(f"  ✅ Layer {layer_idx} remapped")

    # --- Embedding (direct copy) ---
    if "model.embed_tokens.weight" in state_dict:
        new_state_dict["embedding.weight"] = state_dict["model.embed_tokens.weight"]
    
    # --- Final layernorm / LM head (check what keys exist) ---
    if "model.norm.weight" in state_dict:
        new_state_dict["final_norm.scale"] = state_dict["model.norm.weight"].to(torch.float32)
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
    from architecture.gptoss import Transformer, ModelConfig

    model_path = Path(model_path)
    if config is None:
        config_file = model_path / "config.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                config_dict = json.load(f)
        # Filter to only keys ModelConfig accepts
        import inspect
        valid_keys = inspect.signature(ModelConfig.__init__).parameters.keys()
        filtered_config = {k: v for k, v in config_dict.items() if k in valid_keys}
        print(f"Using config keys: {list(filtered_config.keys())}")
        print(f"Skipped keys: {[k for k in config_dict if k not in valid_keys]}")
        config = ModelConfig(**filtered_config)
    else:
        raise ValueError("No config.json found.")
    # Init model
    model = Transformer(config, device=device)
    # Load sharded checkpoint
    print("Loading sharded checkpoint...")
    raw_state_dict = load_sharded_state_dict(str(model_path), device=device)

    # Remap keys + dequantize
    print("Remapping keys and dequantizing experts...")
    state_dict = remap_state_dict(raw_state_dict)

    # Debug: check for mismatches before loading
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(state_dict.keys())
    missing = model_keys - checkpoint_keys
    unexpected = checkpoint_keys - model_keys

    if missing:
        print(f"\n⚠️  Missing keys ({len(missing)}):")
        for k in sorted(missing):
            print(f"    {k}")
    if unexpected:
        print(f"\n⚠️  Unexpected keys ({len(unexpected)}):")
        for k in sorted(unexpected):
            print(f"    {k}")

    # Load
    model.load_state_dict(state_dict, strict=strict)
    print(f"\n✅ Successfully loaded GPT-OSS model")
    return model