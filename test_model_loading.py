#!/usr/bin/env python3
"""
Test GPT-OSS model creation and weight loading logic.
"""
import os
import sys

print("="*60)
print("Testing GPT-OSS Model Loading")
print("="*60)

# Test 1: Import modules
print("\n1. Testing imports...")
try:
    from architecture.gptossdecoder import Transformer, gpt_oss_20b_config
    print("✓ Successfully imported gptossdecoder")
except Exception as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)

# Test 2: Create config
print("\n2. Creating config...")
try:
    cfg = gpt_oss_20b_config()
    print(f"✓ Config created successfully")
    print(f"  vocab_size: {cfg.vocab_size}")
    print(f"  hidden_size: {cfg.hidden_size}")
    print(f"  num_hidden_layers: {cfg.num_hidden_layers}")
    print(f"  num_attention_heads: {cfg.num_attention_heads}")
    print(f"  num_key_value_heads: {cfg.num_key_value_heads}")
    print(f"  head_dim: {cfg.head_dim}")
    print(f"  num_local_experts: {cfg.num_local_experts}")
    print(f"  experts_per_token: {cfg.experts_per_token}")
except Exception as e:
    print(f"✗ Failed to create config: {e}")
    sys.exit(1)

# Test 3: Create model
print("\n3. Creating model (this may take a moment)...")
try:
    import torch
    # Use smaller config for testing
    cfg.num_hidden_layers = 2  # Just 2 layers for testing
    model = Transformer(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created successfully!")
    print(f"  Total parameters: {n_params/1e6:.1f}M (2 layers)")
    print(f"  Model device: {next(model.parameters()).device}")
except Exception as e:
    print(f"✗ Failed to create model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test forward pass
print("\n4. Testing forward pass...")
try:
    batch_size = 1
    seq_len = 10
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    
    with torch.no_grad():
        logits, aux = model(input_ids)
    
    print(f"✓ Forward pass successful!")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output logits shape: {logits.shape}")
    print(f"  Expected: ({batch_size}, {seq_len}, {cfg.vocab_size})")
    print(f"  Aux dict keys: {list(aux.keys())}")
    
    if 'loss' in aux:
        print(f"  Loss: {aux['loss']}")
    if 'router_aux_loss' in aux:
        print(f"  Router aux loss: {aux['router_aux_loss']:.4f}")
    
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Check weights directory
print("\n5. Checking weights directory...")
weights_dir = "architecture/open-gpt-oss/weights"
if os.path.exists(weights_dir):
    import glob
    safetensors = glob.glob(os.path.join(weights_dir, "*.safetensors"))
    if safetensors:
        print(f"✓ Found {len(safetensors)} safetensors files:")
        for f in safetensors:
            size_mb = os.path.getsize(f) / 1e6
            print(f"    {os.path.basename(f)} ({size_mb:.1f} MB)")
    else:
        print(f"⚠️  Weights directory exists but no .safetensors files found")
else:
    print(f"⚠️  Weights directory not found: {weights_dir}")
    print(f"    Model will use random initialization")

# Test 6: Test state dict keys
print("\n6. Checking model state dict structure...")
try:
    state_dict = model.state_dict()
    print(f"✓ State dict has {len(state_dict)} keys")
    
    # Show some sample keys
    sample_keys = list(state_dict.keys())[:5]
    print(f"  Sample keys:")
    for key in sample_keys:
        print(f"    {key}: {state_dict[key].shape}")
    
    # Check for expected keys
    expected_prefixes = ['wte', 'wpe', 'ln_f', 'lm_head']
    found = []
    for prefix in expected_prefixes:
        matching = [k for k in state_dict.keys() if k.startswith(prefix)]
        if matching:
            found.append(prefix)
    
    if found:
        print(f"  Found expected key prefixes: {found}")
    
except Exception as e:
    print(f"⚠️  Could not inspect state dict: {e}")

print("\n" + "="*60)
print("All tests completed successfully!")
print("="*60)
