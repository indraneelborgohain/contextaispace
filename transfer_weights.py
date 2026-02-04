"""
Weight transfer script: HuggingFace GPT-OSS → Custom GPToss model.

Maps HuggingFace naming convention to custom model naming convention
and copies weights directly.
"""

import gc
import random
import torch
from transformers import AutoModelForCausalLM
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
    output_path: str = "gptoss_custom_weights.pt",
    device: str = "cpu",
    verify: bool = True,
):
    """
    Transfer weights from HuggingFace model to custom GPToss model.
    
    Args:
        hf_model_name: HuggingFace model identifier or local path
        output_path: Path to save the converted weights
        device: Device to load models on (use "cpu" for memory efficiency)
        verify: Whether to verify weight transfer with random sampling
    
    Returns:
        custom_model: The custom model with transferred weights
    """
    
    # ─── 1. Load HF model ───
    print("Loading HuggingFace model...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    hf_state = hf_model.state_dict()
    print(f"  HF model loaded with {len(hf_state)} weight tensors")
    
    # ─── 2. Create custom model (empty) ───
    print("\nCreating custom model...")
    custom_model = Transformer(ModelConfig)
    custom_state = custom_model.state_dict()
    print(f"  Custom model created with {len(custom_state)} weight tensors")
    
    # ─── 3. Transfer weights ───
    print("\nTransferring weights...")
    transferred = 0
    skipped = []
    shape_mismatches = []
    not_in_custom = []
    
    for hf_key, hf_tensor in hf_state.items():
        custom_key = map_hf_to_custom(hf_key)
        
        if custom_key is None:
            skipped.append(hf_key)
            continue
        
        if custom_key not in custom_state:
            not_in_custom.append((hf_key, custom_key))
            continue
        
        # Shape check
        if hf_tensor.shape != custom_state[custom_key].shape:
            shape_mismatches.append({
                'hf_key': hf_key,
                'custom_key': custom_key,
                'hf_shape': hf_tensor.shape,
                'custom_shape': custom_state[custom_key].shape
            })
            continue
        
        # Copy weight
        custom_state[custom_key].copy_(hf_tensor)
        transferred += 1
    
    # ─── 4. Load weights into model ───
    custom_model.load_state_dict(custom_state)
    
    # ─── 5. Summary ───
    print("\n" + "=" * 80)
    print("WEIGHT TRANSFER SUMMARY")
    print("=" * 80)
    print(f"✅ Transferred:  {transferred}/{len(hf_state)} weights")
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
    
    # ─── 6. Verify weights ───
    if verify:
        print("\n" + "=" * 80)
        print("VERIFICATION (sampling 5 random weights)")
        print("=" * 80)
        
        # Only sample from successfully transferred keys
        transferred_hf_keys = [
            k for k in hf_state.keys() 
            if map_hf_to_custom(k) is not None 
            and map_hf_to_custom(k) in custom_state
        ]
        
        sample_keys = random.sample(transferred_hf_keys, min(5, len(transferred_hf_keys)))
        all_match = True
        
        for hf_key in sample_keys:
            custom_key = map_hf_to_custom(hf_key)
            hf_val = hf_state[hf_key].float().mean().item()
            custom_val = custom_state[custom_key].float().mean().item()
            match = abs(hf_val - custom_val) < 1e-6
            status = "✅" if match else "❌"
            all_match = all_match and match
            
            print(f"{status} {hf_key}")
            print(f"   → {custom_key}")
            print(f"   HF mean: {hf_val:.8f}, Custom mean: {custom_val:.8f}")
        
        if all_match:
            print("\n✅ All sampled weights verified successfully!")
        else:
            print("\n❌ Some weights did not match!")
    
    # ─── 7. Save custom model ───
    print(f"\nSaving custom model to '{output_path}'...")
    torch.save(custom_model.state_dict(), output_path)
    print("✅ Done!")
    
    # ─── 8. Clean up HF model ───
    del hf_model, hf_state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return custom_model


def load_custom_model(weights_path: str = "gptoss_custom_weights.pt", device: str = "cuda:0"):
    """
    Load custom model from saved weights.
    
    Args:
        weights_path: Path to the saved weights file
        device: Device to load model on
    
    Returns:
        model: Loaded custom model ready for inference
    """
    print(f"Loading custom model from '{weights_path}'...")
    model = Transformer(ModelConfig)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"✅ Model loaded to {device}")
    return model


if __name__ == "__main__":
    # Run weight transfer using local model files
    custom_model = transfer_weights(
        hf_model_name="model/gpt-oss-20b",  # Use local path
        output_path="model/gptoss_custom_weights.pt",
        device="cpu",  # Use CPU to fit both models in memory
        verify=True,
    )
    
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("1. Check that 'Transferred' count matches expected (e.g., 411 weights)")
    print("2. Verify all sampled weights show ✅")
    print("3. Run inference with your custom model:")
    print("")
    print("   from transfer_weights import load_custom_model")
    print("   model = load_custom_model('gptoss_custom_weights.pt', device='cuda:0')")
    print("   # Then use your generate_text function")
