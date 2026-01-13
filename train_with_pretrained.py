#!/usr/bin/env python3
"""
train_with_pretrained.py - Fine-tune trained encoder-decoder with new learning rates

This script allows you to:
1. Load your trained encoder-decoder checkpoint
2. Fine-tune with new (typically lower) learning rates
3. Useful for continued training or domain adaptation
"""
import argparse
import json
import math
import os
import time
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset

from architecture.config import ModelConfig
from architecture.encoder import BidirectionalEncoder, EncoderConfig
from architecture.transformer import Transformer
from architecture.tokenizer import get_tokenizer

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None


def get_args():
    ap = argparse.ArgumentParser(
        description="Fine-tune trained encoder-decoder with new learning rates"
    )
    
    # Model paths
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="Optional: Path to checkpoint with trained encoder and decoder (.pt file)")
    ap.add_argument("--decoder_checkpoint", type=str, default=None,
                    help="Optional: Path to checkpoint with just decoder weights (.pt file)")
    ap.add_argument("--gptoss_weights", type=str, default=None,
                    help="Optional: Directory with GPT-OSS weights (will load compatible decoder layers)")
    ap.add_argument("--gpt2_model", type=str, default=None,
                    help="Optional: GPT-2 model name (e.g., 'gpt2', 'gpt2-medium', 'gpt2-large') for decoder")
    ap.add_argument("--bert_model", type=str, default=None,
                    help="Optional: BERT model name (e.g., 'bert-base-uncased', 'roberta-base') for encoder")
    ap.add_argument("--max_decoder_layers", type=int, default=None,
                    help="Optional: Limit number of decoder layers (useful for loading partial pretrained weights)")
    ap.add_argument("--out_dir", type=str, default="model_finetuned")
    
    # Model size
    ap.add_argument("--model_size", type=str, 
                    choices=["toy", "small", "medium", "large"], default="small")
    
    # Training hyperparameters
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_context_len", type=int, default=512)
    ap.add_argument("--max_qa_len", type=int, default=128)
    ap.add_argument("--max_iters", type=int, default=5000)
    ap.add_argument("--log_interval", type=int, default=10)
    ap.add_argument("--eval_interval", type=int, default=200)
    ap.add_argument("--eval_iters", type=int, default=20)
    ap.add_argument("--save_every", type=int, default=500)
    
    # Learning rates
    ap.add_argument("--encoder_lr", type=float, default=1e-5,
                    help="Learning rate for encoder")
    ap.add_argument("--decoder_lr", type=float, default=1e-5,
                    help="Learning rate for decoder (use 1e-6 if loading GPT-OSS)")
    ap.add_argument("--cross_attn_lr", type=float, default=3e-4,
                    help="Learning rate for custom layers (context proj, cross-attn)")
    
    # Optimizer
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--warmup_iters", type=int, default=200)
    ap.add_argument("--min_lr_ratio", type=float, default=0.1,
                    help="Minimum LR as ratio of initial LR")
    
    # System
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--dtype", type=str, 
                    choices=["float32", "bfloat16", "float16"], default="bfloat16")
    
    # Dataset
    ap.add_argument("--dataset", type=str, default="msmarco",
                    choices=["msmarco", "squad"])
    
    # TensorBoard
    ap.add_argument("--use_tensorboard", action="store_true", default=False)
    ap.add_argument("--log_dir", type=str, default="runs_pretrained")
    
    return ap.parse_args()


def load_bert_weights_partial(encoder, bert_model_name, device):
    """
    Load BERT/RoBERTa weights for compatible encoder layers.
    Keeps custom SVD compression and cross-attention layers.
    
    Args:
        encoder: BidirectionalEncoder instance
        bert_model_name: Hugging Face model name (e.g., 'bert-base-uncased', 'roberta-base')
        device: torch device
    
    Returns:
        Number of parameters loaded
    """
    try:
        from transformers import AutoModel
    except ImportError:
        print("⚠️  transformers not installed. Install with: pip install transformers")
        return 0
    
    print(f"\nLoading BERT weights from: {bert_model_name}")
    
    try:
        # Load pretrained BERT model
        bert_model = AutoModel.from_pretrained(bert_model_name)
        bert_state = bert_model.state_dict()
        print(f"✓ Downloaded BERT model ({len(bert_state)} parameters)")
    except Exception as e:
        print(f"❌ Failed to load BERT: {e}")
        return 0
    
    # Get encoder state
    encoder_state = encoder.state_dict()
    loaded_count = 0
    
    # 1. Load embedding
    if 'embeddings.word_embeddings.weight' in bert_state:
        if encoder_state['embedding.weight'].shape == bert_state['embeddings.word_embeddings.weight'].shape:
            encoder_state['embedding.weight'] = bert_state['embeddings.word_embeddings.weight'].to(device)
            loaded_count += 1
            print(f"  ✓ Loaded embedding.weight")
    
    # 2. Load each encoder block
    num_layers = encoder.config.num_hidden_layers
    for layer_idx in range(num_layers):
        bert_prefix = f'encoder.layer.{layer_idx}'
        enc_prefix = f'blocks.{layer_idx}'
        
        # Check if this layer exists in BERT
        if f'{bert_prefix}.attention.self.query.weight' not in bert_state:
            continue
        
        # Load attention QKV (BERT has separate Q,K,V, we need to concat them)
        try:
            q_weight = bert_state[f'{bert_prefix}.attention.self.query.weight']
            k_weight = bert_state[f'{bert_prefix}.attention.self.key.weight']
            v_weight = bert_state[f'{bert_prefix}.attention.self.value.weight']
            
            q_bias = bert_state[f'{bert_prefix}.attention.self.query.bias']
            k_bias = bert_state[f'{bert_prefix}.attention.self.key.bias']
            v_bias = bert_state[f'{bert_prefix}.attention.self.value.bias']
            
            # Concatenate Q, K, V
            qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
            qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
            
            if encoder_state[f'{enc_prefix}.attn.qkv.weight'].shape == qkv_weight.shape:
                encoder_state[f'{enc_prefix}.attn.qkv.weight'] = qkv_weight.to(device)
                encoder_state[f'{enc_prefix}.attn.qkv.bias'] = qkv_bias.to(device)
                loaded_count += 2
                print(f"  ✓ Loaded {enc_prefix}.attn.qkv")
        except Exception as e:
            print(f"  ⚠️  Skipped {enc_prefix}.attn.qkv: {e}")
        
        # Load attention output
        try:
            if f'{bert_prefix}.attention.output.dense.weight' in bert_state:
                out_weight = bert_state[f'{bert_prefix}.attention.output.dense.weight']
                out_bias = bert_state[f'{bert_prefix}.attention.output.dense.bias']
                
                if encoder_state[f'{enc_prefix}.attn.out.weight'].shape == out_weight.shape:
                    encoder_state[f'{enc_prefix}.attn.out.weight'] = out_weight.to(device)
                    encoder_state[f'{enc_prefix}.attn.out.bias'] = out_bias.to(device)
                    loaded_count += 2
                    print(f"  ✓ Loaded {enc_prefix}.attn.out")
        except Exception as e:
            print(f"  ⚠️  Skipped {enc_prefix}.attn.out: {e}")
        
        # Load FFN (MLP)
        try:
            if f'{bert_prefix}.intermediate.dense.weight' in bert_state:
                fc1_weight = bert_state[f'{bert_prefix}.intermediate.dense.weight']
                fc1_bias = bert_state[f'{bert_prefix}.intermediate.dense.bias']
                
                if encoder_state[f'{enc_prefix}.mlp.fc1.weight'].shape == fc1_weight.shape:
                    encoder_state[f'{enc_prefix}.mlp.fc1.weight'] = fc1_weight.to(device)
                    encoder_state[f'{enc_prefix}.mlp.fc1.bias'] = fc1_bias.to(device)
                    loaded_count += 2
                    print(f"  ✓ Loaded {enc_prefix}.mlp.fc1")
        except Exception as e:
            print(f"  ⚠️  Skipped {enc_prefix}.mlp.fc1: {e}")
        
        try:
            if f'{bert_prefix}.output.dense.weight' in bert_state:
                fc2_weight = bert_state[f'{bert_prefix}.output.dense.weight']
                fc2_bias = bert_state[f'{bert_prefix}.output.dense.bias']
                
                if encoder_state[f'{enc_prefix}.mlp.fc2.weight'].shape == fc2_weight.shape:
                    encoder_state[f'{enc_prefix}.mlp.fc2.weight'] = fc2_weight.to(device)
                    encoder_state[f'{enc_prefix}.mlp.fc2.bias'] = fc2_bias.to(device)
                    loaded_count += 2
                    print(f"  ✓ Loaded {enc_prefix}.mlp.fc2")
        except Exception as e:
            print(f"  ⚠️  Skipped {enc_prefix}.mlp.fc2: {e}")
    
    # Load the updated state
    if loaded_count > 0:
        encoder.load_state_dict(encoder_state)
        print(f"\n✓ Loaded {loaded_count} parameters from BERT")
    else:
        print(f"\n⚠️  No matching parameters found")
    
    print(f"✓ Total encoder parameters: {sum(p.numel() for p in encoder.parameters())/1e6:.2f}M")
    
    return loaded_count


def load_gpt2_weights_partial(decoder, gpt2_model_name, device, max_layers=None):
    """
    Load GPT-2 pretrained weights into decoder (only compatible base layers).
    Skips custom context layers. Works with GPT-2 small/medium/large/xl.
    
    Args:
        decoder: Transformer instance
        gpt2_model_name: GPT-2 model name ('gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl')
        device: torch device
        max_layers: Optional limit on number of layers to load
    
    Returns:
        Number of parameters loaded
    """
    try:
        from transformers import GPT2LMHeadModel
    except ImportError:
        print("⚠️  transformers library required. Install with: pip install transformers")
        return 0
    
    print(f"\nLoading GPT-2 weights from: {gpt2_model_name}")
    
    try:
        gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
        gpt2_state = gpt2_model.state_dict()
        print(f"✓ Loaded GPT-2 state dict ({len(gpt2_state)} parameters)")
    except Exception as e:
        print(f"❌ Failed to load GPT-2 model: {e}")
        return 0
    
    decoder_state = decoder.state_dict()
    loaded_count = 0
    
    print("\nMapping GPT-2 parameters to decoder:")
    
    # Load embedding
    if "transformer.wte.weight" in gpt2_state and "embedding.weight" in decoder_state:
        gpt2_emb = gpt2_state["transformer.wte.weight"]
        dec_emb = decoder_state["embedding.weight"]
        
        # Use minimum vocab size
        min_vocab = min(gpt2_emb.size(0), dec_emb.size(0))
        if gpt2_emb.size(1) == dec_emb.size(1):
            decoder_state["embedding.weight"][:min_vocab] = gpt2_emb[:min_vocab].clone()
            loaded_count += 1
            print(f"  ✓ embedding.weight [{min_vocab}/{dec_emb.size(0)} tokens, {dec_emb.size(1)} dim]")
        else:
            print(f"  ✗ embedding dimension mismatch: GPT-2={gpt2_emb.size(1)}, decoder={dec_emb.size(1)}")
    
    # Determine number of layers to load
    num_gpt2_layers = sum(1 for k in gpt2_state.keys() if k.startswith("transformer.h.") and ".ln_1.weight" in k)
    num_decoder_layers = sum(1 for k in decoder_state.keys() if k.startswith("block.") and ".ln1.weight" in k)
    layers_to_load = min(num_gpt2_layers, num_decoder_layers)
    if max_layers:
        layers_to_load = min(layers_to_load, max_layers)
    
    print(f"  Loading {layers_to_load} layers (GPT-2 has {num_gpt2_layers}, decoder has {num_decoder_layers})")
    
    for layer_idx in range(layers_to_load):
        layer_loaded = 0
        
        # Layer norm 1
        for suffix in ["weight", "bias"]:
            gpt2_key = f"transformer.h.{layer_idx}.ln_1.{suffix}"
            dec_key = f"block.{layer_idx}.ln1.{suffix}"
            if gpt2_key in gpt2_state and dec_key in decoder_state:
                if gpt2_state[gpt2_key].shape == decoder_state[dec_key].shape:
                    decoder_state[dec_key] = gpt2_state[gpt2_key].clone()
                    layer_loaded += 1
        
        # Attention QKV (GPT-2 has c_attn which is fused QKV)
        gpt2_key = f"transformer.h.{layer_idx}.attn.c_attn.weight"
        dec_key = f"block.{layer_idx}.attn.qkv.weight"
        if gpt2_key in gpt2_state and dec_key in decoder_state:
            gpt2_qkv = gpt2_state[gpt2_key]  # [hidden_size, 3*hidden_size]
            dec_qkv = decoder_state[dec_key]  # [3*hidden_size, hidden_size]
            
            # GPT-2 uses [hidden, 3*hidden] (transposed from ours)
            if gpt2_qkv.size(1) == dec_qkv.size(0) and gpt2_qkv.size(0) == dec_qkv.size(1):
                decoder_state[dec_key] = gpt2_qkv.t().clone()  # Transpose
                layer_loaded += 1
        
        gpt2_key = f"transformer.h.{layer_idx}.attn.c_attn.bias"
        dec_key = f"block.{layer_idx}.attn.qkv.bias"
        if gpt2_key in gpt2_state and dec_key in decoder_state:
            if gpt2_state[gpt2_key].shape == decoder_state[dec_key].shape:
                decoder_state[dec_key] = gpt2_state[gpt2_key].clone()
                layer_loaded += 1
        
        # Attention output projection
        gpt2_key = f"transformer.h.{layer_idx}.attn.c_proj.weight"
        dec_key = f"block.{layer_idx}.attn.out.weight"
        if gpt2_key in gpt2_state and dec_key in decoder_state:
            gpt2_out = gpt2_state[gpt2_key]  # [hidden_size, hidden_size]
            dec_out = decoder_state[dec_key]
            if gpt2_out.size(1) == dec_out.size(0) and gpt2_out.size(0) == dec_out.size(1):
                decoder_state[dec_key] = gpt2_out.t().clone()  # Transpose
                layer_loaded += 1
        
        gpt2_key = f"transformer.h.{layer_idx}.attn.c_proj.bias"
        dec_key = f"block.{layer_idx}.attn.out.bias"
        if gpt2_key in gpt2_state and dec_key in decoder_state:
            if gpt2_state[gpt2_key].shape == decoder_state[dec_key].shape:
                decoder_state[dec_key] = gpt2_state[gpt2_key].clone()
                layer_loaded += 1
        
        # Layer norm 2
        for suffix in ["weight", "bias"]:
            gpt2_key = f"transformer.h.{layer_idx}.ln_2.{suffix}"
            dec_key = f"block.{layer_idx}.ln2.{suffix}"
            if gpt2_key in gpt2_state and dec_key in decoder_state:
                if gpt2_state[gpt2_key].shape == decoder_state[dec_key].shape:
                    decoder_state[dec_key] = gpt2_state[gpt2_key].clone()
                    layer_loaded += 1
        
        loaded_count += layer_loaded
        if layer_idx < 3 or layer_idx >= layers_to_load - 1:
            print(f"  ✓ Layer {layer_idx}: {layer_loaded} params")
        elif layer_idx == 3:
            print(f"    ... (loading layers {layer_idx} to {layers_to_load-1})")
    
    # Load final layer norm if present
    if "transformer.ln_f.weight" in gpt2_state and "ln_f.weight" in decoder_state:
        if gpt2_state["transformer.ln_f.weight"].shape == decoder_state["ln_f.weight"].shape:
            decoder_state["ln_f.weight"] = gpt2_state["transformer.ln_f.weight"].clone()
            decoder_state["ln_f.bias"] = gpt2_state["transformer.ln_f.bias"].clone()
            loaded_count += 2
            print(f"  ✓ ln_f (final layer norm)")
    
    print(f"\n✓ Loaded {loaded_count} GPT-2 parameters into decoder")
    print("  Note: Custom layers (context_proj, cross_attn, MoE experts) remain randomly initialized")
    
    # Load the modified state dict
    decoder.load_state_dict(decoder_state, strict=False)
    return loaded_count


def load_gptoss_weights_partial(decoder, gptoss_weights_dir, device, max_layers=None):
    """
    Load GPT-OSS weights for compatible decoder layers.
    Skips custom layers (context_proj, cross_attn).
    
    Args:
        decoder: Transformer instance
        gptoss_weights_dir: Directory with GPT-OSS weights
        device: torch device
    
    Returns:
        Number of parameters loaded
    """
    from pathlib import Path
    
    weights_path = Path(gptoss_weights_dir)
    if not weights_path.exists():
        print(f"❌ GPT-OSS weights not found at {gptoss_weights_dir}")
        return 0
    
    print(f"\nLoading GPT-OSS weights from: {gptoss_weights_dir}")
    
    # Find weights file
    weights_file = None
    for ext in ["*.safetensors", "*.bin", "*.pt", "*.pth"]:
        weight_files = list(weights_path.glob(ext))
        if weight_files:
            weights_file = weight_files[0]
            break
    
    if not weights_file:
        print(f"❌ No weight files found in {gptoss_weights_dir}")
        return 0
    
    print(f"Loading from: {weights_file.name}")
    
    # Load weights
    try:
        if weights_file.suffix == ".safetensors":
            try:
                from safetensors.torch import load_file
                gptoss_state = load_file(str(weights_file))
            except ImportError:
                print("⚠️  safetensors not installed. Install with: pip install safetensors")
                return 0
        else:
            gptoss_state = torch.load(weights_file, map_location=device, weights_only=False)
        
        print(f"✓ Loaded GPT-OSS state dict ({len(gptoss_state)} parameters)")
    except Exception as e:
        print(f"❌ Failed to load weights: {e}")
        return 0
    
    # Get decoder state
    decoder_state = decoder.state_dict()
    loaded_count = 0
    
    # Try to figure out GPT-OSS parameter naming by looking at keys
    sample_keys = list(gptoss_state.keys())[:10]
    print(f"Sample GPT-OSS parameters: {sample_keys}")
    
    # Common GPT naming patterns:
    # - transformer.wte.weight (embedding)
    # - transformer.h.N.attn.c_attn.weight (qkv)
    # - transformer.h.N.attn.c_proj.weight (output)
    # - transformer.h.N.mlp.c_fc.weight (fc1)
    # - transformer.h.N.mlp.c_proj.weight (fc2)
    
    # Or simpler:
    # - wte.weight
    # - h.N.attn.qkv.weight
    # - h.N.attn.out.weight
    
    # Determine prefix (transformer. or empty)
    if any('transformer.' in k for k in sample_keys):
        gpt_prefix = 'transformer.'
    else:
        gpt_prefix = ''
    
    # 1. Try to load embedding
    for emb_key in [f'{gpt_prefix}wte.weight', f'{gpt_prefix}token_emb.weight', 'embedding.weight']:
        if emb_key in gptoss_state:
            if decoder_state['embedding.weight'].shape == gptoss_state[emb_key].shape:
                decoder_state['embedding.weight'] = gptoss_state[emb_key].to(device)
                loaded_count += 1
                print(f"  ✓ Loaded embedding from {emb_key}")
                break
    
    # 2. Load each decoder block
    num_layers = decoder.config.num_hidden_layers
    for layer_idx in range(num_layers):
        dec_prefix = f'block.{layer_idx}'
        
        # Try different GPT naming conventions
        for gpt_layer_prefix in [
            f'{gpt_prefix}h.{layer_idx}',
            f'{gpt_prefix}blocks.{layer_idx}',
            f'{gpt_prefix}layers.{layer_idx}',
            f'block.{layer_idx}',
        ]:
            # Try to load attention QKV
            for qkv_key in [f'{gpt_layer_prefix}.attn.c_attn.weight', 
                           f'{gpt_layer_prefix}.attn.qkv.weight',
                           f'{gpt_layer_prefix}.attention.qkv.weight']:
                if qkv_key in gptoss_state:
                    try:
                        qkv_weight = gptoss_state[qkv_key]
                        qkv_bias_key = qkv_key.replace('.weight', '.bias')
                        
                        if decoder_state[f'{dec_prefix}.attn.qkv.weight'].shape == qkv_weight.shape:
                            decoder_state[f'{dec_prefix}.attn.qkv.weight'] = qkv_weight.to(device)
                            loaded_count += 1
                            
                            if qkv_bias_key in gptoss_state:
                                decoder_state[f'{dec_prefix}.attn.qkv.bias'] = gptoss_state[qkv_bias_key].to(device)
                                loaded_count += 1
                            
                            print(f"  ✓ Loaded {dec_prefix}.attn.qkv from {qkv_key}")
                            break
                    except Exception as e:
                        continue
            
            # Try to load attention output
            for out_key in [f'{gpt_layer_prefix}.attn.c_proj.weight',
                           f'{gpt_layer_prefix}.attn.out.weight',
                           f'{gpt_layer_prefix}.attention.out.weight']:
                if out_key in gptoss_state:
                    try:
                        out_weight = gptoss_state[out_key]
                        out_bias_key = out_key.replace('.weight', '.bias')
                        
                        if decoder_state[f'{dec_prefix}.attn.out.weight'].shape == out_weight.shape:
                            decoder_state[f'{dec_prefix}.attn.out.weight'] = out_weight.to(device)
                            loaded_count += 1
                            
                            if out_bias_key in gptoss_state:
                                decoder_state[f'{dec_prefix}.attn.out.bias'] = gptoss_state[out_bias_key].to(device)
                                loaded_count += 1
                            
                            print(f"  ✓ Loaded {dec_prefix}.attn.out from {out_key}")
                            break
                    except Exception as e:
                        continue
            
            # Try to load MLP experts (if GPT-OSS also uses MoE)
            # Skip gate and context_proj - those are custom
            num_experts = decoder.config.num_experts
            for expert_idx in range(num_experts):
                # Try fc1 (first layer of expert)
                for fc1_key in [f'{gpt_layer_prefix}.mlp.experts.{expert_idx}.0.weight',
                               f'{gpt_layer_prefix}.mlp.experts.{expert_idx}.fc1.weight']:
                    if fc1_key in gptoss_state:
                        try:
                            fc1_weight = gptoss_state[fc1_key]
                            fc1_bias_key = fc1_key.replace('.weight', '.bias')
                            
                            if decoder_state[f'{dec_prefix}.mlp.experts.{expert_idx}.0.weight'].shape == fc1_weight.shape:
                                decoder_state[f'{dec_prefix}.mlp.experts.{expert_idx}.0.weight'] = fc1_weight.to(device)
                                loaded_count += 1
                                
                                if fc1_bias_key in gptoss_state:
                                    decoder_state[f'{dec_prefix}.mlp.experts.{expert_idx}.0.bias'] = gptoss_state[fc1_bias_key].to(device)
                                    loaded_count += 1
                                
                                print(f"  ✓ Loaded {dec_prefix}.mlp.experts.{expert_idx}.0")
                                break
                        except Exception:
                            continue
                
                # Try fc2 (second layer of expert)
                for fc2_key in [f'{gpt_layer_prefix}.mlp.experts.{expert_idx}.1.weight',
                               f'{gpt_layer_prefix}.mlp.experts.{expert_idx}.fc2.weight']:
                    if fc2_key in gptoss_state:
                        try:
                            fc2_weight = gptoss_state[fc2_key]
                            fc2_bias_key = fc2_key.replace('.weight', '.bias')
                            
                            if decoder_state[f'{dec_prefix}.mlp.experts.{expert_idx}.1.weight'].shape == fc2_weight.shape:
                                decoder_state[f'{dec_prefix}.mlp.experts.{expert_idx}.1.weight'] = fc2_weight.to(device)
                                loaded_count += 1
                                
                                if fc2_bias_key in gptoss_state:
                                    decoder_state[f'{dec_prefix}.mlp.experts.{expert_idx}.1.bias'] = gptoss_state[fc2_bias_key].to(device)
                                    loaded_count += 1
                                
                                print(f"  ✓ Loaded {dec_prefix}.mlp.experts.{expert_idx}.1")
                                break
                        except Exception:
                            continue
    
    # Load the updated state
    if loaded_count > 0:
        decoder.load_state_dict(decoder_state)
        print(f"\n✓ Loaded {loaded_count} parameters from GPT-OSS")
        print(f"✓ Kept custom layers (context_proj, cross_attn, gate)")
    else:
        print(f"\n⚠️  No matching parameters found between GPT-OSS and decoder")
        print(f"   GPT-OSS may have different architecture or naming")
    
    print(f"✓ Total decoder parameters: {sum(p.numel() for p in decoder.parameters())/1e6:.2f}M")
    
    return loaded_count


def get_lr(it, warmup_iters, max_iters, learning_rate, min_lr):
    """Learning rate schedule with warmup and cosine decay"""
    # Warmup
    if it < warmup_iters:
        return learning_rate * (it + 1) / warmup_iters
    # Cosine decay
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def prepare_optimizer_groups(encoder, decoder, encoder_lr, decoder_lr, cross_attn_lr, 
                             weight_decay, beta1, beta2):
    """
    Create optimizer with different learning rates for:
    1. Pretrained encoder (low LR)
    2. Pretrained decoder (very low LR)
    3. New cross-attention layers (normal LR)
    """
    encoder_params = list(encoder.parameters())
    
    # Separate decoder parameters
    decoder_pretrained_params = []
    cross_attn_params = []
    
    for name, param in decoder.named_parameters():
        if 'cross_attn' in name or 'context_projection' in name:
            cross_attn_params.append(param)
        else:
            decoder_pretrained_params.append(param)
    
    param_groups = [
        {'params': encoder_params, 'lr': encoder_lr, 'name': 'encoder'},
        {'params': decoder_pretrained_params, 'lr': decoder_lr, 'name': 'decoder'},
        {'params': cross_attn_params, 'lr': cross_attn_lr, 'name': 'cross_attn'},
    ]
    
    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(beta1, beta2),
        weight_decay=weight_decay
    )
    
    print(f"\nOptimizer groups:")
    print(f"  - Encoder: {len(encoder_params)} params, lr={encoder_lr:.2e}")
    print(f"  - Decoder (pretrained): {len(decoder_pretrained_params)} params, lr={decoder_lr:.2e}")
    print(f"  - Cross-attention (new): {len(cross_attn_params)} params, lr={cross_attn_lr:.2e}")
    
    return optimizer


def main():
    args = get_args()
    
    # Setup
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Get tokenizer
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.n_vocab
    
    print("="*60)
    print("Training with Pretrained Models")
    print("="*60)
    print(f"Device: {device}")
    print(f"Vocab size: {vocab_size}")
    print(f"Checkpoint: {args.checkpoint if args.checkpoint else 'None (starting fresh)'}")
    print(f"Output directory: {args.out_dir}")
    print("="*60 + "\n")
    
    # Load checkpoint if provided
    checkpoint = None
    if args.checkpoint:
        print(f"Loading models from: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Load encoder config
    from architecture.encoder import BidirectionalEncoder, EncoderConfig
    if checkpoint and 'encoder_config' in checkpoint:
        encoder_config = checkpoint['encoder_config']
        if isinstance(encoder_config, dict):
            encoder_config = EncoderConfig(**encoder_config)
        print(f"✓ Loaded encoder config from checkpoint")
    else:
        # Default config based on model_size
        if args.model_size == "toy":
            encoder_config = EncoderConfig(vocab_size=vocab_size, hidden_size=256, num_hidden_layers=4)
        elif args.model_size == "small":
            encoder_config = EncoderConfig(vocab_size=vocab_size, hidden_size=768, num_hidden_layers=8)
        elif args.model_size == "medium":
            # Note: Use 1024 to match gpt2-medium and bert-large
            encoder_config = EncoderConfig(vocab_size=vocab_size, hidden_size=1024, num_hidden_layers=12)
        else:  # large
            encoder_config = EncoderConfig(vocab_size=vocab_size, hidden_size=1280, num_hidden_layers=16)
        print(f"✓ Created {args.model_size} encoder config")
    
    encoder = BidirectionalEncoder(encoder_config)
    encoder.to(device)
    
    # Only load encoder weights from checkpoint if NOT using BERT
    if checkpoint and not args.bert_model and 'encoder' in checkpoint:
        encoder.load_state_dict(checkpoint['encoder'])
        print(f"✓ Loaded encoder from checkpoint ({sum(p.numel() for p in encoder.parameters())/1e6:.2f}M params)")
    else:
        print(f"✓ Initialized encoder ({sum(p.numel() for p in encoder.parameters())/1e6:.2f}M params)")
    
    # Load decoder config from checkpoint
    if checkpoint and 'decoder_config' in checkpoint:
        decoder_config = checkpoint['decoder_config']
        print(f"Loaded decoder config from checkpoint")
        print(f"  Config type: {type(decoder_config)}")
        
        # If config is a dict, convert to ModelConfig
        if isinstance(decoder_config, dict):
            decoder_config = ModelConfig(**decoder_config)
            print(f"  Converted dict to ModelConfig")
    elif checkpoint and 'config' in checkpoint:
        # Try alternate key
        decoder_config = checkpoint['config']
        if isinstance(decoder_config, dict):
            decoder_config = ModelConfig(**decoder_config)
    else:
        if not checkpoint:
            print("No checkpoint provided, creating fresh decoder config")
        else:
            print("⚠️  No decoder config found in checkpoint, using default")
        
        # Create config based on model_size
        if args.model_size == "toy":
            decoder_config = ModelConfig(
                vocab_size=vocab_size, hidden_size=256, num_hidden_layers=4,
                num_experts=4, num_attention_heads=8, use_encoder_decoder_cross_attention=True
            )
        elif args.model_size == "small":
            decoder_config = ModelConfig(
                vocab_size=vocab_size, hidden_size=768, num_hidden_layers=8,
                num_experts=8, num_attention_heads=16, use_encoder_decoder_cross_attention=True
            )
        elif args.model_size == "medium":
            # Matches GPT-2 medium (1024 hidden, 24 layers)
            decoder_config = ModelConfig(
                vocab_size=vocab_size, hidden_size=1024, num_hidden_layers=24,
                num_experts=32, num_attention_heads=16, num_key_value_heads=8,
                use_encoder_decoder_cross_attention=True
            )
        else:  # large
            # Matches GPT-2 large (1280 hidden, 36 layers)
            decoder_config = ModelConfig(
                vocab_size=vocab_size, hidden_size=1280, num_hidden_layers=36,
                num_experts=32, num_attention_heads=20, num_key_value_heads=8,
                use_encoder_decoder_cross_attention=True
            )
        print(f"✓ Created {args.model_size} decoder config")
    
    # Print config details
    print(f"  Decoder config: {decoder_config.num_hidden_layers} layers, "
          f"{decoder_config.num_experts} experts, "
          f"{decoder_config.num_attention_heads} heads")
    
    decoder = Transformer(decoder_config)
    decoder.to(device)
    
    # Load decoder weights from checkpoint or decoder_checkpoint
    decoder_loaded = False
    
    if args.decoder_checkpoint:
        # Load from separate decoder checkpoint
        print(f"\nLoading decoder from: {args.decoder_checkpoint}")
        decoder_ckpt = torch.load(args.decoder_checkpoint, map_location=device, weights_only=False)
        
        # Handle different checkpoint formats
        if 'decoder' in decoder_ckpt:
            decoder_state = decoder_ckpt['decoder']
        elif 'model' in decoder_ckpt:
            decoder_state = decoder_ckpt['model']
        else:
            # Assume the checkpoint IS the state dict
            decoder_state = decoder_ckpt
        
        try:
            decoder.load_state_dict(decoder_state, strict=False)
            print(f"✓ Loaded decoder from {args.decoder_checkpoint} ({sum(p.numel() for p in decoder.parameters())/1e6:.2f}M params)")
            decoder_loaded = True
        except RuntimeError as e:
            print(f"❌ Error loading decoder from {args.decoder_checkpoint}: {str(e)[:200]}")
            print(f"⚠️  Continuing with randomly initialized decoder")
    
    elif checkpoint and 'decoder' in checkpoint:
        # Load from full checkpoint
        try:
            decoder.load_state_dict(checkpoint['decoder'], strict=True)
            print(f"✓ Loaded decoder from checkpoint ({sum(p.numel() for p in decoder.parameters())/1e6:.2f}M params)")
            decoder_loaded = True
        except RuntimeError as e:
            print(f"❌ Error loading decoder weights:")
            print(f"   {str(e)[:200]}...")
            print(f"\n⚠️  This usually means the checkpoint config doesn't match")
            print(f"   Try checking what's in the checkpoint with:")
            print(f"   python -c \"import torch; ckpt=torch.load('{args.checkpoint}', weights_only=False); print(ckpt.keys())\"")
            raise
    
    if not decoder_loaded:
        print(f"✓ Initialized decoder ({sum(p.numel() for p in decoder.parameters())/1e6:.2f}M params)")
    
    # Optionally load BERT weights for encoder
    if args.bert_model:
        print("\n" + "="*60)
        print("Hybrid Encoder Loa ding: BERT + Trained Model")
        print("="*60)
        print("Strategy:")
        print("  - BERT weights: embedding, attention, FFN")
        print("  - Trained weights: SVD compression, cross-attention")
        print()
        
        loaded = load_bert_weights_partial(encoder, args.bert_model, device)
        
        if loaded > 0:
            print(f"\n✓ Hybrid encoder created successfully!")
            print(f"  BERT provides pretrained language understanding")
            print(f"  Your model provides SVD compression & cross-attention")
        else:
            print(f"\n⚠️  No BERT weights loaded, using only trained encoder")
        print("="*60 + "\n")
    
    # Optionally load GPT-2 weights for decoder (preferred for single GPU)
    if args.gpt2_model:
        print("\n" + "="*60)
        print("Hybrid Decoder Loading: GPT-2 + Trained Model")
        print("="*60)
        print("Strategy:")
        print("  - GPT-2 weights: embedding, attention, layer norms")
        print("  - Trained weights: context projections, cross-attention, MoE experts")
        print()
        
        loaded = load_gpt2_weights_partial(decoder, args.gpt2_model, device, max_layers=args.max_decoder_layers)
        
        if loaded > 0:
            print(f"\n✓ Hybrid decoder created successfully!")
            print(f"  GPT-2 provides pretrained language modeling")
            print(f"  Your model provides context-awareness & MoE")
        else:
            print(f"\n⚠️  No GPT-2 weights loaded, using random initialization")
        print("="*60 + "\n")
    
    # Optionally load GPT-OSS weights for compatible layers (overwrites base layers)
    elif args.gptoss_weights:
        print("\n" + "="*60)
        print("Hybrid Decoder Loading: GPT-OSS + Trained Model")
        print("="*60)
        print("Strategy:")
        print("  - GPT-OSS weights: embedding, standard attention, MLP (overwrite)")
        print("  - Trained weights: context projections, cross-attention (keep)")
        print()
        
        loaded = load_gptoss_weights_partial(decoder, args.gptoss_weights, device)
        
        if loaded > 0:
            print(f"\n✓ Hybrid model created successfully!")
            print(f"  GPT-OSS provides base transformer knowledge")
            print(f"  Your trained model provides custom functionality")
        else:
            print(f"\n⚠️  No GPT-OSS weights loaded, using only trained model")
        print("="*60 + "\n")
    
    # Setup optimizer with different learning rates
    encoder_params = list(encoder.parameters())
    
    # Separate decoder parameters into base (potentially from GPT-OSS) and custom
    decoder_base_params = []
    decoder_custom_params = []
    
    for name, param in decoder.named_parameters():
        if any(custom in name for custom in [
            'context_proj',
            'cross_attn', 
            'start_token_embedding',
            'lsi',
            'encoder_projection'
        ]):
            decoder_custom_params.append(param)
        else:
            decoder_base_params.append(param)
    
    # Use different LR for custom layers if specified
    if len(decoder_custom_params) > 0 and args.cross_attn_lr != args.decoder_lr:
        optimizer = torch.optim.AdamW([
            {'params': encoder_params, 'lr': args.encoder_lr, 'name': 'encoder'},
            {'params': decoder_base_params, 'lr': args.decoder_lr, 'name': 'decoder_base'},
            {'params': decoder_custom_params, 'lr': args.cross_attn_lr, 'name': 'decoder_custom'},
        ], betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
        
        print(f"\nOptimizer groups (3-tier):")
        print(f"  - Encoder: {len(encoder_params)} params, lr={args.encoder_lr:.2e}")
        print(f"  - Decoder (base): {len(decoder_base_params)} params, lr={args.decoder_lr:.2e}")
        print(f"  - Decoder (custom): {len(decoder_custom_params)} params, lr={args.cross_attn_lr:.2e}")
        use_three_tier = True
    else:
        optimizer = torch.optim.AdamW([
            {'params': encoder_params, 'lr': args.encoder_lr, 'name': 'encoder'},
            {'params': decoder_base_params + decoder_custom_params, 'lr': args.decoder_lr, 'name': 'decoder'},
        ], betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
        
        print(f"\nOptimizer groups (2-tier):")
        print(f"  - Encoder: {len(encoder_params)} params, lr={args.encoder_lr:.2e}")
        print(f"  - Decoder: {len(decoder_base_params) + len(decoder_custom_params)} params, lr={args.decoder_lr:.2e}")
        use_three_tier = False
    
    # TensorBoard
    writer = None
    if args.use_tensorboard and TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(args.log_dir)
        print(f"TensorBoard logging to: {args.log_dir}")
    
    # Load dataset
    print(f"\nLoading {args.dataset} dataset...")
    if args.dataset == "msmarco":
        # Load MS MARCO
        from dataloader.data_loader import load_msmarco_data
        train_examples, val_examples = load_msmarco_data(
            tokenizer, args.max_context_len, args.max_qa_len
        )
    else:
        # Load SQuAD
        from dataloader.data_loader import load_squad_data
        train_examples, val_examples = load_squad_data(
            tokenizer, args.max_context_len, args.max_qa_len
        )
    
    print(f"Training examples: {len(train_examples)}")
    print(f"Validation examples: {len(val_examples)}")
    
    # Set dtype context
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype_ctx = dtype_map[args.dtype]
    
    # Training loop
    print("\n" + "="*60)
    print("Starting training...")
    print(f"Max iterations: {args.max_iters}")
    print("="*60 + "\n")
    
    encoder.train()
    decoder.train()
    
    iter_num = 0
    running_loss = 0.0
    log_count = 0
    best_val_loss = float('inf')
    
    t0 = time.time()
    
    while iter_num < args.max_iters:
        # Get batch (implement your data loading here)
        # This is a placeholder - replace with your actual data loading
        batch = get_training_batch(train_examples, args.batch_size, device)
        
        # Update learning rates with schedule
        encoder_lr = get_lr(iter_num, args.warmup_iters, args.max_iters, 
                           args.encoder_lr, args.encoder_lr * args.min_lr_ratio)
        decoder_lr = get_lr(iter_num, args.warmup_iters, args.max_iters,
                           args.decoder_lr, args.decoder_lr * args.min_lr_ratio)
        
        optimizer.param_groups[0]['lr'] = encoder_lr
        optimizer.param_groups[1]['lr'] = decoder_lr
        
        if use_three_tier:
            custom_lr = get_lr(iter_num, args.warmup_iters, args.max_iters,
                              args.cross_attn_lr, args.cross_attn_lr * args.min_lr_ratio)
            optimizer.param_groups[2]['lr'] = custom_lr
        
        # Forward pass
        # TODO: Implement your forward pass here
        # This should encode context, decode with cross-attention, and compute loss
        
        # Placeholder loss computation
        loss = torch.tensor(0.0, device=device)
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()),
                args.grad_clip
            )
        
        optimizer.step()
        
        # Logging
        running_loss += loss.item()
        log_count += 1
        
        if (iter_num + 1) % args.log_interval == 0:
            avg_loss = running_loss / log_count
            t1 = time.time()
            dt = t1 - t0
            
            if use_three_tier:
                custom_lr = optimizer.param_groups[2]['lr']
                print(f"iter {iter_num + 1:5d} | loss {avg_loss:.4f} | "
                      f"enc_lr {encoder_lr:.2e} | dec_lr {decoder_lr:.2e} | "
                      f"custom_lr {custom_lr:.2e} | {dt*1000:.2f}ms")
            else:
                print(f"iter {iter_num + 1:5d} | loss {avg_loss:.4f} | "
                      f"enc_lr {encoder_lr:.2e} | dec_lr {decoder_lr:.2e} | {dt*1000:.2f}ms")
            
            if writer:
                writer.add_scalar('Loss/train', avg_loss, iter_num + 1)
                writer.add_scalar('LR/encoder', encoder_lr, iter_num + 1)
                writer.add_scalar('LR/decoder', decoder_lr, iter_num + 1)
                if use_three_tier:
                    writer.add_scalar('LR/custom', custom_lr, iter_num + 1)
            
            running_loss = 0.0
            log_count = 0
            t0 = time.time()
        
        # Save checkpoint
        if (iter_num + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.out_dir, f"checkpoint_{iter_num + 1}.pt")
            torch.save({
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'encoder_config': encoder_config,
                'optimizer': optimizer.state_dict(),
                'iter': iter_num + 1,
                'args': vars(args),
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
        
        iter_num += 1
    
    # Save final model
    final_path = os.path.join(args.out_dir, "final_model.pt")
    torch.save({
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'encoder_config': encoder_config,
    }, final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")
    
    if writer:
        writer.close()


def get_training_batch(examples, batch_size, device):
    """
    Placeholder function - implement your actual batch creation here
    """
    # TODO: Implement based on your data format
    return {}


if __name__ == "__main__":
    main()