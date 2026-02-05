"""
Weight transfer script: HuggingFace GPT-OSS → Custom GPToss model.

Maps HuggingFace naming convention to custom model naming convention
and copies weights directly.

Memory-optimized: Uses streaming weight transfer to avoid loading
both full models into memory simultaneously.
"""

import gc
import os
import random
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from glob import glob
from transformers import AutoModelForCausalLM, AutoConfig
from architecture.gptoss20B import Transformer, ModelConfig


def map_hf_to_custom(hf_key: str) -> str:
    """Convert HuggingFace key to custom model key."""
    
    # Embedding
    if hf_key == "model.embed_tokens.weight":
        return "embed.weight"
    
    # Final norm
    if hf_key == "model.norm.weight":
        return "norm_f.weight"
    
    # LM head (if present)
    if hf_key == "lm_head.weight":
        return "lm_head.weight"
    
    # Layer-specific mappings
    if hf_key.startswith("model.layers."):
        parts = hf_key.split(".")
        layer_idx = parts[2]
        rest = ".".join(parts[3:])
        
        mappings = {
            "input_layernorm.weight": f"layers.{layer_idx}.norm1.weight",
            "post_attention_layernorm.weight": f"layers.{layer_idx}.norm2.weight",
            "self_attn.sinks": f"layers.{layer_idx}.attn.sink_logit",
            "self_attn.q_proj.weight": f"layers.{layer_idx}.attn.q.weight",
            "self_attn.q_proj.bias": f"layers.{layer_idx}.attn.q.bias",
            "self_attn.k_proj.weight": f"layers.{layer_idx}.attn.k.weight",
            "self_attn.k_proj.bias": f"layers.{layer_idx}.attn.k.bias",
            "self_attn.v_proj.weight": f"layers.{layer_idx}.attn.v.weight",
            "self_attn.v_proj.bias": f"layers.{layer_idx}.attn.v.bias",
            "self_attn.o_proj.weight": f"layers.{layer_idx}.attn.o.weight",
            "self_attn.o_proj.bias": f"layers.{layer_idx}.attn.o.bias",
            "mlp.experts.gate_up_proj": f"layers.{layer_idx}.moe.W_in",
            "mlp.experts.gate_up_proj_bias": f"layers.{layer_idx}.moe.b_in",
            "mlp.experts.down_proj": f"layers.{layer_idx}.moe.W_out",
            "mlp.experts.down_proj_bias": f"layers.{layer_idx}.moe.b_out",
            "mlp.router.weight": f"layers.{layer_idx}.moe.router.weight",
            "mlp.router.bias": f"layers.{layer_idx}.moe.router.bias",
        }
        
        if rest in mappings:
            return mappings[rest]
    
    return None  # Unmapped key


def transfer_weights(
    hf_model_name: str = "openai/gpt-oss-20b",
    output_path: str = "gptoss_custom_weights.safetensors",
    device: str = "cpu",
    verify: bool = True,
    use_safetensors_direct: bool = True,
):
    """
    Transfer weights from HuggingFace model to custom GPToss model.
    
    Memory-optimized version with two modes:
    - use_safetensors_direct=True: Streams from .safetensors files (lowest memory)
    - use_safetensors_direct=False: Uses AutoModelForCausalLM with memory optimizations
    
    Args:
        hf_model_name: HuggingFace model identifier or local path
        output_path: Path to save the converted weights (.safetensors)
        device: Device to load models on (use "cpu" for memory efficiency)
        verify: Whether to verify weight transfer with random sampling
        use_safetensors_direct: If True, read directly from safetensors files.
                                If False, use AutoModelForCausalLM API.
    
    Returns:
        custom_model: The custom model with transferred weights
    """
    
    if use_safetensors_direct:
        return _transfer_weights_safetensors(hf_model_name, output_path, device, verify)
    else:
        return _transfer_weights_hf_api(hf_model_name, output_path, device, verify)


def _transfer_weights_hf_api(
    hf_model_name: str,
    output_path: str,
    device: str,
    verify: bool,
):
    """Transfer weights using AutoModelForCausalLM with memory optimizations."""
    
    # ─── 1. Load HF model with memory optimizations ───
    print("Loading HuggingFace model (memory-optimized)...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": device},  # Load to specific device
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    print(f"  HF model loaded")
    
    # ─── 2. Get custom model expected keys (meta device = no memory) ───
    print("\nCreating custom model structure...")
    with torch.device("meta"):
        meta_model = Transformer(ModelConfig)
    custom_keys = set(meta_model.state_dict().keys())
    custom_shapes = {k: v.shape for k, v in meta_model.state_dict().items()}
    del meta_model
    print(f"  Custom model expects {len(custom_keys)} weight tensors")
    
    # ─── 3. Stream weights from HF model ───
    print("\nTransferring weights...")
    transferred_weights = {}
    transferred = 0
    skipped = []
    shape_mismatches = []
    not_in_custom = []
    verification_samples = []
    total_keys = 0
    
    # Iterate through HF state dict WITHOUT making a copy
    for hf_key, hf_tensor in hf_model.state_dict().items():
        total_keys += 1
        custom_key = map_hf_to_custom(hf_key)
        
        if custom_key is None:
            skipped.append(hf_key)
            continue
        
        if custom_key not in custom_keys:
            not_in_custom.append((hf_key, custom_key))
            continue
        
        # Shape check
        expected_shape = custom_shapes[custom_key]
        if hf_tensor.shape != expected_shape:
            shape_mismatches.append({
                'hf_key': hf_key,
                'custom_key': custom_key,
                'hf_shape': hf_tensor.shape,
                'custom_shape': expected_shape
            })
            continue
        
        # Clone and store weight (detach from HF model graph)
        transferred_weights[custom_key] = hf_tensor.clone().to(torch.bfloat16)
        transferred += 1
        
        # Collect samples for verification
        if verify and len(verification_samples) < 5:
            verification_samples.append({
                'hf_key': hf_key,
                'custom_key': custom_key,
                'hf_mean': hf_tensor.float().mean().item()
            })
    
    # ─── 4. Delete HF model BEFORE creating custom model ───
    print("\nFreeing HuggingFace model from memory...")
    del hf_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # ─── 5. Summary ───
    print("\n" + "=" * 80)
    print("WEIGHT TRANSFER SUMMARY")
    print("=" * 80)
    print(f"✅ Transferred:  {transferred}/{total_keys} weights")
    print(f"⚠️  Skipped (unmapped): {len(skipped)} keys")
    print(f"❌ Not in custom model: {len(not_in_custom)} keys")
    print(f"❌ Shape mismatches: {len(shape_mismatches)} keys")
    
    if skipped:
        print("\nSkipped keys (first 10):")
        for key in skipped[:10]:
            print(f"   {key}")
    
    if not_in_custom:
        print("\nMapped but not in custom model:")
        for hf_key, custom_key in not_in_custom[:10]:
            print(f"   {hf_key} → {custom_key}")
    
    if shape_mismatches:
        print("\nShape mismatches:")
        for mismatch in shape_mismatches[:10]:
            print(f"   {mismatch['custom_key']}:")
            print(f"      HF: {mismatch['hf_shape']}, Custom: {mismatch['custom_shape']}")
    
    # ─── 6. Save weights as safetensors ───
    print(f"\nSaving custom weights to '{output_path}'...")
    save_file(transferred_weights, output_path)
    print(f"✅ Saved {len(transferred_weights)} weights")
    
    # ─── 7. Clear transferred weights from memory ───
    del transferred_weights
    gc.collect()
    
    # ─── 8. Load custom model with transferred weights ───
    print("\nLoading custom model with new weights...")
    custom_model = Transformer(ModelConfig)
    
    with safe_open(output_path, framework="pt", device=device) as f:
        state_dict = {key: f.get_tensor(key) for key in f.keys()}
    
    custom_model.load_state_dict(state_dict, strict=False)
    del state_dict
    gc.collect()
    
    # ─── 9. Verify weights ───
    if verify and verification_samples:
        print("\n" + "=" * 80)
        print("VERIFICATION (sampling weights)")
        print("=" * 80)
        
        all_match = True
        custom_state = custom_model.state_dict()
        
        for sample in verification_samples:
            custom_val = custom_state[sample['custom_key']].float().mean().item()
            match = abs(sample['hf_mean'] - custom_val) < 1e-4
            status = "✅" if match else "❌"
            all_match = all_match and match
            
            print(f"{status} {sample['hf_key']}")
            print(f"   → {sample['custom_key']}")
            print(f"   HF mean: {sample['hf_mean']:.8f}, Custom mean: {custom_val:.8f}")
        
        del custom_state
        
        if all_match:
            print("\n✅ All sampled weights verified successfully!")
        else:
            print("\n❌ Some weights did not match!")
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("\n✅ Transfer complete!")
    return custom_model


def _transfer_weights_safetensors(
    hf_model_name: str,
    output_path: str,
    device: str,
    verify: bool,
):
    """
    Transfer weights by streaming directly from safetensors files.
    Lowest memory usage - never loads full HF model.
    """
    
    # ─── 1. Find safetensors files ───
    print("Locating weight files...")
    safetensor_files = sorted(glob(os.path.join(hf_model_name, "*.safetensors")))
    
    if not safetensor_files:
        raise FileNotFoundError(
            f"No .safetensors files found in {hf_model_name}. "
            "Please ensure the model is downloaded with safetensors format."
        )
    
    print(f"  Found {len(safetensor_files)} safetensor file(s)")
    
    # ─── 2. Build key index (memory-light scan) ───
    print("\nScanning weight keys...")
    hf_key_to_file = {}
    total_keys = 0
    
    for sf_path in safetensor_files:
        with safe_open(sf_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                hf_key_to_file[key] = sf_path
                total_keys += 1
    
    print(f"  Indexed {total_keys} weight tensors")
    
    # ─── 3. Create custom model (meta device = no memory) ───
    print("\nCreating custom model structure...")
    with torch.device("meta"):
        custom_model = Transformer(ModelConfig)
    
    custom_keys = set(custom_model.state_dict().keys())
    print(f"  Custom model expects {len(custom_keys)} weight tensors")
    
    # ─── 4. Stream and transfer weights ───
    print("\nTransferring weights (streaming)...")
    transferred_weights = {}
    transferred = 0
    skipped = []
    shape_mismatches = []
    not_in_custom = []
    verification_samples = []
    
    # Process each safetensor file one at a time
    for sf_path in safetensor_files:
        filename = os.path.basename(sf_path)
        print(f"  Processing {filename}...")
        
        with safe_open(sf_path, framework="pt", device=device) as f:
            for hf_key in f.keys():
                custom_key = map_hf_to_custom(hf_key)
                
                if custom_key is None:
                    skipped.append(hf_key)
                    continue
                
                if custom_key not in custom_keys:
                    not_in_custom.append((hf_key, custom_key))
                    continue
                
                # Load tensor on-demand
                hf_tensor = f.get_tensor(hf_key)
                
                # Get expected shape from meta model
                expected_shape = custom_model.state_dict()[custom_key].shape
                
                # Shape check
                if hf_tensor.shape != expected_shape:
                    shape_mismatches.append({
                        'hf_key': hf_key,
                        'custom_key': custom_key,
                        'hf_shape': hf_tensor.shape,
                        'custom_shape': expected_shape
                    })
                    del hf_tensor
                    continue
                
                # Store weight (convert to bfloat16 for memory efficiency)
                transferred_weights[custom_key] = hf_tensor.to(torch.bfloat16)
                transferred += 1
                
                # Collect samples for verification
                if verify and len(verification_samples) < 5:
                    verification_samples.append({
                        'hf_key': hf_key,
                        'custom_key': custom_key,
                        'hf_mean': hf_tensor.float().mean().item()
                    })
                
                del hf_tensor
        
        # Force garbage collection after each file
        gc.collect()
    
    # ─── 5. Summary ───
    print("\n" + "=" * 80)
    print("WEIGHT TRANSFER SUMMARY")
    print("=" * 80)
    print(f"✅ Transferred:  {transferred}/{total_keys} weights")
    print(f"⚠️  Skipped (unmapped): {len(skipped)} keys")
    print(f"❌ Not in custom model: {len(not_in_custom)} keys")
    print(f"❌ Shape mismatches: {len(shape_mismatches)} keys")
    
    if skipped:
        print("\nSkipped keys (first 10):")
        for key in skipped[:10]:
            print(f"   {key}")
    
    if not_in_custom:
        print("\nMapped but not in custom model:")
        for hf_key, custom_key in not_in_custom[:10]:
            print(f"   {hf_key} → {custom_key}")
    
    if shape_mismatches:
        print("\nShape mismatches:")
        for mismatch in shape_mismatches[:10]:
            print(f"   {mismatch['custom_key']}:")
            print(f"      HF: {mismatch['hf_shape']}, Custom: {mismatch['custom_shape']}")
    
    # ─── 6. Save weights as safetensors ───
    print(f"\nSaving custom weights to '{output_path}'...")
    save_file(transferred_weights, output_path)
    print(f"✅ Saved {len(transferred_weights)} weights")
    
    # ─── 7. Clear transferred weights from memory ───
    del transferred_weights
    gc.collect()
    
    # ─── 8. Load custom model with transferred weights ───
    print("\nLoading custom model with new weights...")
    custom_model = Transformer(ModelConfig)
    
    with safe_open(output_path, framework="pt", device=device) as f:
        state_dict = {key: f.get_tensor(key) for key in f.keys()}
    
    custom_model.load_state_dict(state_dict, strict=False)
    del state_dict
    gc.collect()
    
    # ─── 9. Verify weights ───
    if verify and verification_samples:
        print("\n" + "=" * 80)
        print("VERIFICATION (sampling weights)")
        print("=" * 80)
        
        all_match = True
        custom_state = custom_model.state_dict()
        
        for sample in verification_samples:
            custom_val = custom_state[sample['custom_key']].float().mean().item()
            match = abs(sample['hf_mean'] - custom_val) < 1e-4  # bfloat16 tolerance
            status = "✅" if match else "❌"
            all_match = all_match and match
            
            print(f"{status} {sample['hf_key']}")
            print(f"   → {sample['custom_key']}")
            print(f"   HF mean: {sample['hf_mean']:.8f}, Custom mean: {custom_val:.8f}")
        
        del custom_state
        
        if all_match:
            print("\n✅ All sampled weights verified successfully!")
        else:
            print("\n❌ Some weights did not match!")
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("\n✅ Transfer complete!")
    return custom_model


def load_custom_model(weights_path: str = "gptoss_custom_weights.safetensors", device: str = "cuda:0"):
    """
    Load custom model from saved weights.
    
    Args:
        weights_path: Path to the saved weights file (.safetensors or .pt)
        device: Device to load model on
    
    Returns:
        model: Loaded custom model ready for inference
    """
    print(f"Loading custom model from '{weights_path}'...")
    model = Transformer(ModelConfig)
    
    if weights_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(weights_path, device=device)
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(torch.load(weights_path, map_location=device))
    
    model = model.to(device)
    model.eval()
    print(f"✅ Model loaded to {device}")
    return model


if __name__ == "__main__":
    # Run weight transfer using local model files
    # Set use_safetensors_direct=False to use AutoModelForCausalLM instead
    custom_model = transfer_weights(
        hf_model_name="model/gpt-oss-20b",  # Use local path or HuggingFace ID
        output_path="model/gptoss_custom_weights.safetensors",
        device="cpu",  # Use CPU to minimize memory
        verify=True,
        use_safetensors_direct=True,  # False = use AutoModelForCausalLM
    )
    
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("1. Check that 'Transferred' count matches expected (e.g., 411 weights)")
    print("2. Verify all sampled weights show ✅")
    print("3. Run inference with your custom model:")
    print("")
    print("   from transfer_weights import load_custom_model")
    print("   model = load_custom_model('model/gptoss_custom_weights.safetensors', device='cuda:0')")
    print("   # Then use your generate_text function")
