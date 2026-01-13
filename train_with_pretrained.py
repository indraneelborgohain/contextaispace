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
    ap.add_argument("--checkpoint", type=str, required=True,
                    help="Path to checkpoint with trained encoder and decoder (.pt file)")
    ap.add_argument("--gptoss_weights", type=str, default=None,
                    help="Optional: Directory with GPT-OSS weights (will load compatible decoder layers)")
    ap.add_argument("--bert_model", type=str, default=None,
                    help="Optional: BERT model name (e.g., 'bert-base-uncased', 'roberta-base') for encoder")
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
        print(f"✓ Downloaded BERT model")
    except Exception as e:
        print(f"❌ Failed to load BERT: {e}")
        return 0
    
    # Get encoder state
    encoder_state = encoder.state_dict()
    
    # Mapping between BERT and our encoder
    # BERT uses: embeddings.word_embeddings, encoder.layer.N.attention, etc.
    # We use: embedding, blocks.N.attn, etc.
    
    loaded_count = 0
    skipped_count = 0
    
    for name, param in encoder_state.items():
        # Skip custom compression and cross-attention layers
        if any(skip in name for skip in [
            'cross_attn',
            'final_cross_attn',
            'lsi',
            'compression',
            '_compress'
        ]):
            skipped_count += 1
            continue
        
        # Try to map our parameter names to BERT names
        bert_name = None
        
        # Embedding layer
        if name == 'embedding.weight':
            bert_name = 'embeddings.word_embeddings.weight'
        
        # Encoder blocks: blocks.N.attn.qkv.weight -> encoder.layer.N.attention.self.query/key/value.weight
        elif 'blocks.' in name:
            parts = name.split('.')
            layer_idx = parts[1]
            
            if 'attn.qkv.weight' in name:
                # BERT splits QKV, we might need to concat them
                # This is complex, skip for now and handle separately
                pass
            elif 'attn.out.weight' in name:
                bert_name = f'encoder.layer.{layer_idx}.attention.output.dense.weight'
            elif 'attn.norm' in name:
                bert_name = f'encoder.layer.{layer_idx}.attention.output.LayerNorm.weight'
            elif 'mlp' in name or 'ffn' in name:
                # Map FFN layers
                if 'mlp.0.weight' in name or 'experts.0.0.weight' in name:
                    bert_name = f'encoder.layer.{layer_idx}.intermediate.dense.weight'
                elif 'mlp.1.weight' in name or 'experts.0.1.weight' in name:
                    bert_name = f'encoder.layer.{layer_idx}.output.dense.weight'
        
        # Norm layer
        elif name == 'norm.weight':
            bert_name = 'encoder.layer.11.output.LayerNorm.weight'  # Last layer norm
        
        # Try to load if we found a mapping
        if bert_name and bert_name in bert_state:
            bert_param = bert_state[bert_name]
            if param.shape == bert_param.shape:
                encoder_state[name] = bert_param.to(device)
                loaded_count += 1
            else:
                # Shape mismatch, keep original
                pass
    
    # Load the updated state
    encoder.load_state_dict(encoder_state)
    
    print(f"✓ Loaded {loaded_count} parameters from BERT")
    print(f"✓ Kept {skipped_count} custom parameters (SVD, cross-attn)")
    print(f"✓ Total encoder parameters: {sum(p.numel() for p in encoder.parameters())/1e6:.2f}M")
    
    return loaded_count


def load_gptoss_weights_partial(decoder, gptoss_weights_dir, device):
    """Load GPT-OSS weights for compatible layers only"""
    from pathlib import Path
    
    weights_path = Path(gptoss_weights_dir)
    if not weights_path.exists():
        print(f"GPT-OSS weights not found at {gptoss_weights_dir}, skipping")
        return 0
    
    print(f"\nLoading GPT-OSS weights from: {gptoss_weights_dir}")
    
    # Find weights file
    weights_file = None
    for ext in ["*.safetensors", "*.bin", "*.pt"]:
        weight_files = list(weights_path.glob(ext))
        if weight_files:
            weights_file = weight_files[0]
            break
    
    if not weights_file:
        print("No weight files found, skipping GPT-OSS loading")
        return 0
    
    print(f"Loading from: {weights_file.name}")
    
    # Load weights
    if weights_file.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
            gptoss_state = load_file(str(weights_file))
        except ImportError:
            print("safetensors not installed, skipping")
            return 0
    else:
        gptoss_state = torch.load(weights_file, map_location=device, weights_only=False)
    
    # Get decoder state
    decoder_state = decoder.state_dict()
    
    # Try to load compatible weights
    loaded_count = 0
    skipped_count = 0
    
    for name, param in decoder_state.items():
        # Try exact match first
        if name in gptoss_state:
            gptoss_param = gptoss_state[name]
            if param.shape == gptoss_param.shape:
                decoder_state[name] = gptoss_param.to(device)
                loaded_count += 1
                continue
        
        # Try common mappings (GPT-OSS might use different names)
        # embedding.weight -> embedding.weight
        # block.0.attn.qkv.weight -> block.0.attn.qkv.weight
        # etc.
        
        # Skip context-specific layers (will keep trained weights)
        if any(skip in name for skip in [
            'context_proj',
            'cross_attn',
            'context_state',
            'start_token_embedding',
            'lsi',
            'encoder_projection'
        ]):
            skipped_count += 1
            continue
    
    # Load the updated state
    decoder.load_state_dict(decoder_state)
    
    print(f"✓ Loaded {loaded_count} parameters from GPT-OSS")
    print(f"✓ Kept {skipped_count} custom parameters from trained model")
    print(f"✓ Total parameters: {sum(p.numel() for p in decoder.parameters())/1e6:.2f}M")
    
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
    vocab_size = tokenizer.vocab_size
    
    print("="*60)
    print("Training with Pretrained Models")
    print("="*60)
    print(f"Device: {device}")
    print(f"Vocab size: {vocab_size}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output directory: {args.out_dir}")
    print("="*60 + "\n")
    
    # Load both encoder and decoder from checkpoint
    print(f"Loading models from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Load encoder
    from architecture.encoder import BidirectionalEncoder, EncoderConfig
    if 'encoder_config' in checkpoint:
        encoder_config = checkpoint['encoder_config']
    else:
        # Default config if not found
        encoder_config = EncoderConfig(
            vocab_size=vocab_size,
            hidden_size=1024,
            num_hidden_layers=8,
        )
    
    encoder = BidirectionalEncoder(encoder_config)
    encoder.to(device)
    encoder.load_state_dict(checkpoint['encoder'])
    print(f"✓ Loaded encoder ({sum(p.numel() for p in encoder.parameters())/1e6:.2f}M params)")
    
    # Load decoder
    if 'decoder_config' in checkpoint:
        decoder_config = checkpoint['decoder_config']
    else:
        # Default config if not found
        decoder_config = ModelConfig(
            vocab_size=vocab_size,
            hidden_size=encoder_config.hidden_size,
            num_hidden_layers=8,
            use_encoder_decoder_cross_attention=True,
        )
    
    decoder = Transformer(decoder_config)
    decoder.to(device)
    decoder.load_state_dict(checkpoint['decoder'])
    print(f"✓ Loaded dBERT weights for encoder
    if args.bert_model:
        print("\n" + "="*60)
        print("Hybrid Encoder Loading: BERT + Trained Model")
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
    
    # Optionally load GPT-OSS weights for compatible decodern decoder.parameters())/1e6:.2f}M params)")
    
    # Optionally load GPT-OSS weights for compatible layers
    if args.gptoss_weights:
        print("\n" + "="*60)
        print("Hybrid Loading: GPT-OSS + Trained Model")
        print("="*60)
        print("Strategy:")
        print("  - GPT-OSS weights: embedding, standard attention, MLP")
        print("  - Trained weights: context projections, cross-attention")
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
Fine-Tuning Encoder-Decoder with New Learning Rate