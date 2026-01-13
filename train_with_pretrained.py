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
    ap.add_argument("--gptoss_weights", type=str, default="architecture/open-gpt-oss/weights",
                    help="Optional: Directory with GPT-OSS weights (will load compatible decoder layers)")
    ap.add_argument("--gpt2_model", type=str, default=None,
                    help="Optional: GPT-2 model name (e.g., 'gpt2', 'gpt2-medium', 'gpt2-large') for decoder")
    ap.add_argument("--bert_model", type=str, default="bert-large-uncased",
                    help="Optional: BERT model name (e.g., 'bert-base-uncased', 'roberta-base') for encoder")
    ap.add_argument("--max_decoder_layers", type=int, default=12,
                    help="Limit number of decoder layers (default: 12 for GPU-friendly GPT-OSS loading)")
    ap.add_argument("--out_dir", type=str, default="model_finetuned")
    
    # Model size
    ap.add_argument("--model_size", type=str, 
                    choices=["toy", "small", "medium", "large"], default="medium")
    
    # Training hyperparameters
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_context_len", type=int, default=512)
    ap.add_argument("--max_qa_len", type=int, default=128)
    ap.add_argument("--max_iters", type=int, default=150000)
    ap.add_argument("--log_interval", type=int, default=10)
    ap.add_argument("--eval_interval", type=int, default=200)
    ap.add_argument("--eval_iters", type=int, default=20)
    ap.add_argument("--save_every", type=int, default=500)
    
    # Learning rates
    ap.add_argument("--encoder_lr", type=float, default=1e-5,
                    help="Learning rate for encoder")
    ap.add_argument("--decoder_lr", type=float, default=1e-6,
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
    
    print(f"Encoder expects {len([k for k in encoder_state.keys() if 'blocks' in k])} block parameters")
    print(f"BERT has {len([k for k in bert_state.keys() if 'encoder.layer' in k])} layer parameters")
    
    # 1. Load embedding
    if 'embeddings.word_embeddings.weight' in bert_state:
        bert_emb = bert_state['embeddings.word_embeddings.weight']
        enc_emb = encoder_state['embedding.weight']
        print(f"  BERT embedding: {bert_emb.shape}, Encoder embedding: {enc_emb.shape}")
        
        if enc_emb.shape == bert_emb.shape:
            encoder_state['embedding.weight'] = bert_emb.to(device)
            loaded_count += 1
    
    # 2. Load each encoder block
    num_bert_layers = sum(1 for k in bert_state.keys() if k.startswith('encoder.layer.') and '.attention.self.query.weight' in k)
    num_enc_layers = encoder.config.num_hidden_layers
    layers_to_load = min(num_bert_layers, num_enc_layers)
    
    for layer_idx in range(layers_to_load):
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
        except Exception as e:
            pass
        
        # Load attention output
        try:
            if f'{bert_prefix}.attention.output.dense.weight' in bert_state:
                out_weight = bert_state[f'{bert_prefix}.attention.output.dense.weight']
                out_bias = bert_state[f'{bert_prefix}.attention.output.dense.bias']
                
                if encoder_state[f'{enc_prefix}.attn.out.weight'].shape == out_weight.shape:
                    encoder_state[f'{enc_prefix}.attn.out.weight'] = out_weight.to(device)
                    encoder_state[f'{enc_prefix}.attn.out.bias'] = out_bias.to(device)
                    loaded_count += 2
        except Exception as e:
            pass
        
        # Load FFN (MLP)
        try:
            if f'{bert_prefix}.intermediate.dense.weight' in bert_state:
                fc1_weight = bert_state[f'{bert_prefix}.intermediate.dense.weight']
                fc1_bias = bert_state[f'{bert_prefix}.intermediate.dense.bias']
                
                if encoder_state[f'{enc_prefix}.mlp.fc1.weight'].shape == fc1_weight.shape:
                    encoder_state[f'{enc_prefix}.mlp.fc1.weight'] = fc1_weight.to(device)
                    encoder_state[f'{enc_prefix}.mlp.fc1.bias'] = fc1_bias.to(device)
                    loaded_count += 2
        except Exception as e:
            pass
        
        try:
            if f'{bert_prefix}.output.dense.weight' in bert_state:
                fc2_weight = bert_state[f'{bert_prefix}.output.dense.weight']
                fc2_bias = bert_state[f'{bert_prefix}.output.dense.bias']
                
                if encoder_state[f'{enc_prefix}.mlp.fc2.weight'].shape == fc2_weight.shape:
                    encoder_state[f'{enc_prefix}.mlp.fc2.weight'] = fc2_weight.to(device)
                    encoder_state[f'{enc_prefix}.mlp.fc2.bias'] = fc2_bias.to(device)
                    loaded_count += 2
        except Exception as e:
            pass
    
    # Load the updated state by directly copying parameter data (preserves device)
    if loaded_count > 0:
        for name, param in encoder.named_parameters():
            if name in encoder_state:
                param.data.copy_(encoder_state[name])
        for name, buf in encoder.named_buffers():
            if name in encoder_state:
                buf.data.copy_(encoder_state[name])
        print("Encoder loaded")
    
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
    except Exception as e:
        return 0
    
    decoder_state = decoder.state_dict()
    loaded_count = 0
    
    # Load embedding
    if "transformer.wte.weight" in gpt2_state and "embedding.weight" in decoder_state:
        gpt2_emb = gpt2_state["transformer.wte.weight"]
        dec_emb = decoder_state["embedding.weight"]
        
        # Use minimum vocab size
        min_vocab = min(gpt2_emb.size(0), dec_emb.size(0))
        if gpt2_emb.size(1) == dec_emb.size(1):
            decoder_state["embedding.weight"][:min_vocab] = gpt2_emb[:min_vocab].clone()
            loaded_count += 1
    
    # Determine number of layers to load
    num_gpt2_layers = sum(1 for k in gpt2_state.keys() if k.startswith("transformer.h.") and ".ln_1.weight" in k)
    num_decoder_layers = sum(1 for k in decoder_state.keys() if k.startswith("block.") and ".ln1.weight" in k)
    layers_to_load = min(num_gpt2_layers, num_decoder_layers)
    if max_layers:
        layers_to_load = min(layers_to_load, max_layers)
    
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
    
    # Load final layer norm if present
    if "transformer.ln_f.weight" in gpt2_state and "ln_f.weight" in decoder_state:
        if gpt2_state["transformer.ln_f.weight"].shape == decoder_state["ln_f.weight"].shape:
            decoder_state["ln_f.weight"] = gpt2_state["transformer.ln_f.weight"].clone()
            decoder_state["ln_f.bias"] = gpt2_state["transformer.ln_f.bias"].clone()
            loaded_count += 2
    
    # Load the modified state by directly copying parameter data (preserves device)
    for name, param in decoder.named_parameters():
        if name in decoder_state:
            param.data.copy_(decoder_state[name])
    for name, buf in decoder.named_buffers():
        if name in decoder_state:
            buf.data.copy_(decoder_state[name])
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
    
    # Find ALL weight files (GPT-OSS is sharded across multiple files)
    weight_files = []
    for ext in ["*.safetensors", "*.bin", "*.pt", "*.pth"]:
        found_files = sorted(list(weights_path.glob(ext)))
        if found_files:
            weight_files = found_files
            break
    
    if not weight_files:
        print(f"❌ No weight files found in {gptoss_weights_dir}")
        return 0
    
    print(f"Found {len(weight_files)} weight file(s)")
    if len(weight_files) > 1:
        print(f"  Loading sharded weights: {weight_files[0].name} ... {weight_files[-1].name}")
    else:
        print(f"  Loading from: {weight_files[0].name}")
    
    # Load ALL weight files (handle sharded models)
    gptoss_state = {}
    try:
        for weight_file in weight_files:
            if weight_file.suffix == ".safetensors":
                try:
                    from safetensors.torch import load_file
                    shard_state = load_file(str(weight_file))
                    gptoss_state.update(shard_state)
                except ImportError:
                    print("⚠️  safetensors not installed. Install with: pip install safetensors")
                    return 0
            else:
                shard_state = torch.load(weight_file, map_location=device, weights_only=False)
                gptoss_state.update(shard_state)
    except Exception as e:
        return 0
    
    # Get decoder state
    decoder_state = decoder.state_dict()
    loaded_count = 0
    
    # Try to figure out GPT-OSS parameter naming by looking at keys
    sample_keys = list(gptoss_state.keys())[:10]
    
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
                break
    
    # 2. Load each decoder block
    num_layers = decoder.config.num_hidden_layers
    if max_layers:
        num_layers = min(num_layers, max_layers)
    
    for layer_idx in range(num_layers):
        dec_prefix = f'block.{layer_idx}'
        layer_loaded = False
        
        # Try different GPT naming conventions
        for gpt_layer_prefix in [
            f'{gpt_prefix}h.{layer_idx}',
            f'{gpt_prefix}blocks.{layer_idx}',
            f'{gpt_prefix}layers.{layer_idx}',
            f'layers.{layer_idx}',  # GPT-OSS actual format
            f'block.{layer_idx}',
        ]:
            if layer_loaded:
                break
                
            # Option 1: Try fused QKV (GPT-2 style)
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
                            
                            layer_loaded = True
                            break
                    except Exception as e:
                        continue
            
            # Option 2: Try separate Q, K, V (GPT-OSS style)
            if not layer_loaded:
                q_key = f'{gpt_layer_prefix}.attn.q.weight'
                k_key = f'{gpt_layer_prefix}.attn.k.weight'
                v_key = f'{gpt_layer_prefix}.attn.v.weight'
                
                if q_key in gptoss_state and k_key in gptoss_state and v_key in gptoss_state:
                    try:
                        q_weight = gptoss_state[q_key]
                        k_weight = gptoss_state[k_key]
                        v_weight = gptoss_state[v_key]
                        
                        expected_qkv = decoder_state[f'{dec_prefix}.attn.qkv.weight']
                        
                        # Concatenate Q, K, V
                        qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
                        
                        if decoder_state[f'{dec_prefix}.attn.qkv.weight'].shape == qkv_weight.shape:
                            decoder_state[f'{dec_prefix}.attn.qkv.weight'] = qkv_weight.to(device)
                            loaded_count += 1
                            
                            # Try to load biases if they exist
                            q_bias_key = f'{gpt_layer_prefix}.attn.q.bias'
                            k_bias_key = f'{gpt_layer_prefix}.attn.k.bias'
                            v_bias_key = f'{gpt_layer_prefix}.attn.v.bias'
                            
                            if q_bias_key in gptoss_state and k_bias_key in gptoss_state and v_bias_key in gptoss_state:
                                q_bias = gptoss_state[q_bias_key]
                                k_bias = gptoss_state[k_bias_key]
                                v_bias = gptoss_state[v_bias_key]
                                qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
                                decoder_state[f'{dec_prefix}.attn.qkv.bias'] = qkv_bias.to(device)
                                loaded_count += 1
                            
                            layer_loaded = True
                    except Exception as e:
                        continue
            
            # Try to load attention output
            for out_key in [f'{gpt_layer_prefix}.attn.c_proj.weight',
                           f'{gpt_layer_prefix}.attn.out.weight',
                           f'{gpt_layer_prefix}.attn.o.weight',  # GPT-OSS actual format
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
                            
                            break
                    except Exception as e:
                        continue
            
            # Try to load MoE/MLP weights
            # GPT-OSS uses: layers.N.moe.W_in, layers.N.moe.W_out (single MoE, not per-expert)
            # Try loading GPT-OSS MoE format first
            moe_w_in_key = f'{gpt_layer_prefix}.moe.W_in'
            moe_w_out_key = f'{gpt_layer_prefix}.moe.W_out'
            
            if moe_w_in_key in gptoss_state and moe_w_out_key in gptoss_state:
                # GPT-OSS has MoE structure - we need to distribute to our experts
                # For now, skip this as architectures don't match
                continue
            
            # Try standard MLP (for non-MoE models)
            mlp_fc1_key = f'{gpt_layer_prefix}.mlp.fc1.weight'
            mlp_fc2_key = f'{gpt_layer_prefix}.mlp.fc2.weight'
            
            if mlp_fc1_key in gptoss_state:
                # Standard MLP - load into first expert
                try:
                    fc1_weight = gptoss_state[mlp_fc1_key]
                    fc1_bias = gptoss_state.get(f'{gpt_layer_prefix}.mlp.fc1.bias')
                    
                    if decoder_state[f'{dec_prefix}.mlp.experts.0.0.weight'].shape == fc1_weight.shape:
                        decoder_state[f'{dec_prefix}.mlp.experts.0.0.weight'] = fc1_weight.to(device)
                        loaded_count += 1
                        if fc1_bias is not None:
                            decoder_state[f'{dec_prefix}.mlp.experts.0.0.bias'] = fc1_bias.to(device)
                            loaded_count += 1
                except Exception as e:
                    pass
                    
                try:
                    fc2_weight = gptoss_state[mlp_fc2_key]
                    fc2_bias = gptoss_state.get(f'{gpt_layer_prefix}.mlp.fc2.bias')
                    
                    if decoder_state[f'{dec_prefix}.mlp.experts.0.1.weight'].shape == fc2_weight.shape:
                        decoder_state[f'{dec_prefix}.mlp.experts.0.1.weight'] = fc2_weight.to(device)
                        loaded_count += 1
                        if fc2_bias is not None:
                            decoder_state[f'{dec_prefix}.mlp.experts.0.1.bias'] = fc2_bias.to(device)
                            loaded_count += 1
                except Exception as e:
                    pass
    
    # Load the updated state by directly copying parameter data (preserves device)
    if loaded_count > 0:
        for name, param in decoder.named_parameters():
            if name in decoder_state:
                param.data.copy_(decoder_state[name])
        for name, buf in decoder.named_buffers():
            if name in decoder_state:
                buf.data.copy_(decoder_state[name])
        print("Decoder loaded")
    
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
            encoder_config = EncoderConfig(vocab_size=vocab_size, hidden_size=256, num_hidden_layers=4,
                                          num_attention_heads=8, num_key_value_heads=8)
        elif args.model_size == "small":
            # Matches bert-base (768 hidden, 12 heads)
            encoder_config = EncoderConfig(vocab_size=vocab_size, hidden_size=768, num_hidden_layers=8,
                                          num_attention_heads=12, num_key_value_heads=12)
        elif args.model_size == "medium":
            # Use 1024 (bert-large compatible) - cross-attention handles dimension mismatch with decoder
            encoder_config = EncoderConfig(vocab_size=vocab_size, hidden_size=1024, num_hidden_layers=12,
                                          num_attention_heads=16, num_key_value_heads=16)
        else:  # large
            encoder_config = EncoderConfig(vocab_size=vocab_size, hidden_size=1280, num_hidden_layers=16,
                                          num_attention_heads=20, num_key_value_heads=20)
        print(f"✓ Created {args.model_size} encoder config")
    
    encoder = BidirectionalEncoder(encoder_config, device=device)
    encoder.to(device)
    
    # Only load encoder weights from checkpoint if NOT using BERT
    if checkpoint and not args.bert_model and 'encoder' in checkpoint:
        encoder.load_state_dict(checkpoint['encoder'])
        print(f"✓ Loaded encoder from checkpoint ({sum(p.numel() for p in encoder.parameters())/1e6:.2f}M params)")
    else:
        print(f"✓ Initialized encoder ({sum(p.numel() for p in encoder.parameters())/1e6:.2f}M params)")
    
    # Ensure encoder is on device BEFORE loading any pretrained weights
    encoder.to(device)
    
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
        num_layers = args.max_decoder_layers if args.max_decoder_layers else {"toy": 4, "small": 12, "medium": 24, "large": 36}[args.model_size]
        
        if args.model_size == "toy":
            decoder_config = ModelConfig(
                vocab_size=vocab_size, hidden_size=256, num_hidden_layers=num_layers,
                num_experts=4, num_attention_heads=8, use_encoder_decoder_cross_attention=True
            )
        elif args.model_size == "small":
            # Matches GPT-2 small (768 hidden, 12 layers)
            decoder_config = ModelConfig(
                vocab_size=vocab_size, hidden_size=768, num_hidden_layers=num_layers,
                num_experts=16, num_attention_heads=12, num_key_value_heads=4,
                use_encoder_decoder_cross_attention=True
            )
        elif args.model_size == "medium":
            # Matches GPT-OSS (2880 hidden, 24 layers, but only first 12 loaded by default)
            # Use 64 KV heads (full MHA) to match GPT-OSS - no GQA
            num_layers = args.max_decoder_layers if args.max_decoder_layers else 24
            decoder_config = ModelConfig(
                vocab_size=vocab_size, hidden_size=2880, num_hidden_layers=num_layers,
                num_experts=32, num_attention_heads=64, num_key_value_heads=64,  # Full MHA, not GQA!
                use_encoder_decoder_cross_attention=True,
                encoder_hidden_size=1024  # Encoder uses 1024 (bert-large), decoder uses 2880
            )
        else:  # large
            # Matches GPT-2 large (1280 hidden, 36 layers)
            decoder_config = ModelConfig(
                vocab_size=vocab_size, hidden_size=1280, num_hidden_layers=num_layers,
                num_experts=32, num_attention_heads=20, num_key_value_heads=8,
                use_encoder_decoder_cross_attention=True
            )
        print(f"✓ Created {args.model_size} decoder config")
    
    # Print config details
    print(f"  Decoder config: {decoder_config.num_hidden_layers} layers, "
          f"{decoder_config.num_experts} experts, "
          f"{decoder_config.num_attention_heads} heads")
    if args.max_decoder_layers and args.max_decoder_layers < 24:
        print(f"  📊 Layer limiting enabled: Using {args.max_decoder_layers} layers (reduces memory usage)")
    if args.max_decoder_layers and args.max_decoder_layers < 24:
        print(f"  📊 Layer limiting enabled: Using {args.max_decoder_layers} layers (reduces memory usage)")
    
    decoder = Transformer(decoder_config, device=device)
    # Ensure decoder is on device BEFORE loading any weights
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
        load_bert_weights_partial(encoder, args.bert_model, device)
        encoder = encoder.to(device)
    
    # Optionally load GPT-2 weights for decoder (preferred for single GPU)
    if args.gpt2_model:
        load_gpt2_weights_partial(decoder, args.gpt2_model, device, max_layers=args.max_decoder_layers)
        decoder = decoder.to(device)
    
    # Optionally load GPT-OSS weights for compatible layers (overwrites base layers)
    if args.gptoss_weights:
        load_gptoss_weights_partial(decoder, args.gptoss_weights, device, max_layers=args.max_decoder_layers)
        decoder = decoder.to(device)
    
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
        use_three_tier = True
    else:
        optimizer = torch.optim.AdamW([
            {'params': encoder_params, 'lr': args.encoder_lr, 'name': 'encoder'},
            {'params': decoder_base_params + decoder_custom_params, 'lr': args.decoder_lr, 'name': 'decoder'},
        ], betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
        use_three_tier = False
    
    # TensorBoard
    writer = None
    if args.use_tensorboard and TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(args.log_dir)
        print(f"TensorBoard logging to: {args.log_dir}")
    
    # Load dataset
    print(f"\nLoading {args.dataset} dataset...")
    if args.dataset == "msmarco":
        # Simple MS MARCO loader
        print("Loading MS MARCO dataset...")
        from datasets import load_dataset
        
        dataset = load_dataset("ms_marco", "v2.1")
        
        # Simple formatting - just use query and passages
        train_examples = []
        val_examples = []
        
        print("Formatting training examples...")
        for i, ex in enumerate(dataset['train']):
            if i >= 10000:  # Limit for faster testing
                break
            
            try:
                # MS MARCO structure: passages is a dict with 'passage_text' as a list
                if 'passages' in ex and 'query' in ex:
                    passages = ex['passages']
                    if isinstance(passages, dict) and 'passage_text' in passages:
                        # passages is a dict with lists as values
                        passage_texts = passages['passage_text'][:3] if isinstance(passages['passage_text'], list) else [passages['passage_text']]
                    elif isinstance(passages, list):
                        # passages is a list of dicts
                        passage_texts = [p.get('passage_text', '') for p in passages[:3]]
                    else:
                        continue
                    
                    context = " ".join(passage_texts)
                    question = ex['query']
                    
                    # Get answer
                    if 'answers' in ex and ex['answers']:
                        if isinstance(ex['answers'], list):
                            answer = ex['answers'][0] if ex['answers'] else ""
                        elif isinstance(ex['answers'], dict):
                            answer = ex['answers'].get('text', [''])[0] if 'text' in ex['answers'] else ""
                        else:
                            answer = str(ex['answers'])
                    else:
                        answer = ""
                    
                    # Tokenize
                    ctx_tokens = tokenizer.encode(context)[:args.max_context_len]
                    qa_text = f"{question} {answer}"
                    qa_tokens = tokenizer.encode(qa_text)[:args.max_qa_len]
                    
                    train_examples.append((ctx_tokens, qa_tokens))
            except Exception as e:
                # Skip problematic examples
                if i < 10:  # Only print first few errors
                    print(f"  Warning: Skipping example {i}: {e}")
                continue
        
        print("Formatting validation examples...")
        for i, ex in enumerate(dataset['validation']):
            if i >= 1000:  # Smaller validation set
                break
            
            try:
                if 'passages' in ex and 'query' in ex:
                    passages = ex['passages']
                    if isinstance(passages, dict) and 'passage_text' in passages:
                        passage_texts = passages['passage_text'][:3] if isinstance(passages['passage_text'], list) else [passages['passage_text']]
                    elif isinstance(passages, list):
                        passage_texts = [p.get('passage_text', '') for p in passages[:3]]
                    else:
                        continue
                    
                    context = " ".join(passage_texts)
                    question = ex['query']
                    
                    if 'answers' in ex and ex['answers']:
                        if isinstance(ex['answers'], list):
                            answer = ex['answers'][0] if ex['answers'] else ""
                        elif isinstance(ex['answers'], dict):
                            answer = ex['answers'].get('text', [''])[0] if 'text' in ex['answers'] else ""
                        else:
                            answer = str(ex['answers'])
                    else:
                        answer = ""
                    
                    ctx_tokens = tokenizer.encode(context)[:args.max_context_len]
                    qa_text = f"{question} {answer}"
                    qa_tokens = tokenizer.encode(qa_text)[:args.max_qa_len]
                    
                    val_examples.append((ctx_tokens, qa_tokens))
            except Exception as e:
                if i < 10:
                    print(f"  Warning: Skipping validation example {i}: {e}")
                continue
    else:
        # Simple SQuAD loader
        print("Loading SQuAD dataset...")
        from datasets import load_dataset
        
        dataset = load_dataset("squad")
        
        train_examples = []
        val_examples = []
        
        print("Formatting training examples...")
        for i, ex in enumerate(dataset['train']):
            if i >= 10000:  # Limit for faster testing
                break
            context = ex['context']
            question = ex['question']
            answer = ex['answers']['text'][0] if ex['answers']['text'] else ""
            
            ctx_tokens = tokenizer.encode(context)[:args.max_context_len]
            qa_text = f"{question} {answer}"
            qa_tokens = tokenizer.encode(qa_text)[:args.max_qa_len]
            
            train_examples.append((ctx_tokens, qa_tokens))
        
        print("Formatting validation examples...")
        for i, ex in enumerate(dataset['validation']):
            if i >= 1000:
                break
            context = ex['context']
            question = ex['question']
            answer = ex['answers']['text'][0] if ex['answers']['text'] else ""
            
            ctx_tokens = tokenizer.encode(context)[:args.max_context_len]
            qa_text = f"{question} {answer}"
            qa_tokens = tokenizer.encode(qa_text)[:args.max_qa_len]
            
            val_examples.append((ctx_tokens, qa_tokens))
    
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
    
    # Ensure all components are on correct device (especially RoPE buffers)
    encoder.to(device)
    decoder.to(device)
    
    encoder.train()
    decoder.train()
    
    iter_num = 0
    running_loss = 0.0
    log_count = 0
    best_val_loss = float('inf')
    
    t0 = time.time()
    
    while iter_num < args.max_iters:
        # Get batch
        batch_ctx, batch_qa = get_training_batch(train_examples, args.batch_size, args.max_context_len, args.max_qa_len, device)
        
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
        
        # Zero gradients before accumulation
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass - process each example in batch separately (like other train scripts)
        total_loss = 0.0
        for i in range(batch_ctx.shape[0]):
            ctx_tokens = batch_ctx[i]
            qa_tokens = batch_qa[i]
            
            # Remove padding
            ctx_tokens = ctx_tokens[ctx_tokens != 0]
            qa_mask = qa_tokens != -1
            qa_tokens = qa_tokens[qa_mask]
            
            # Ensure tokens are on correct device
            ctx_tokens = ctx_tokens.to(device)
            qa_tokens = qa_tokens.to(device)
            
            with torch.autocast(device_type='cuda' if 'cuda' in str(device) else 'cpu', dtype=dtype_ctx, enabled=(args.dtype != 'float32')):
                # Encode context and get K,V for cross-attention
                encoder_k, encoder_v = encoder(ctx_tokens, return_encoder_kv=True)
                
                # Decode with cross-attention
                decoder.reset_context()
                logits = decoder(qa_tokens, encoder_k=encoder_k, encoder_v=encoder_v)
                
                # Compute cross-entropy loss (shift by 1 for autoregressive)
                loss = F.cross_entropy(
                    logits[:-1].view(-1, logits.size(-1)),
                    qa_tokens[1:].view(-1),
                    reduction='mean'
                )
            
            # Accumulate loss tensor (like train_msmarco)
            total_loss += loss
            
            # Clear intermediate tensors
            del encoder_k, encoder_v, logits
        
        # Average loss over batch (still a tensor, like train_msmarco)
        loss = total_loss / batch_ctx.shape[0]
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Gradient clipping
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()),
                args.grad_clip
            )
        
        optimizer.step()
        
        # Clear GPU cache periodically
        if (iter_num + 1) % 10 == 0:
            torch.cuda.empty_cache()
        
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


def get_training_batch(examples, batch_size, max_context_len, max_qa_len, device):
    """
    Create a training batch from examples.
    
    Args:
        examples: List of (context_tokens, qa_tokens) tuples
        batch_size: Batch size
        max_context_len: Max context length
        max_qa_len: Max QA length
        device: torch device
    
    Returns:
        batch_ctx: [batch_size, max_context_len] context tokens
        batch_qa: [batch_size, max_qa_len] QA tokens
    """
    import random
    
    # Sample random examples
    batch_examples = random.sample(examples, min(batch_size, len(examples)))
    
    # Pad to max lengths
    batch_ctx = []
    batch_qa = []
    
    for ctx_tokens, qa_tokens in batch_examples:
        # Pad or truncate context
        if len(ctx_tokens) < max_context_len:
            ctx_tokens = ctx_tokens + [0] * (max_context_len - len(ctx_tokens))
        else:
            ctx_tokens = ctx_tokens[:max_context_len]
        
        # Pad or truncate QA
        if len(qa_tokens) < max_qa_len:
            qa_tokens = qa_tokens + [-1] * (max_qa_len - len(qa_tokens))  # Use -1 for padding (will be ignored)
        else:
            qa_tokens = qa_tokens[:max_qa_len]
        
        batch_ctx.append(ctx_tokens)
        batch_qa.append(qa_tokens)
    
    # Convert to tensors
    batch_ctx = torch.tensor(batch_ctx, dtype=torch.long, device=device)
    batch_qa = torch.tensor(batch_qa, dtype=torch.long, device=device)
    
    return batch_ctx, batch_qa


if __name__ == "__main__":
    main()