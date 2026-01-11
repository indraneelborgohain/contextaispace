#!/usr/bin/env python3
"""
train_squad.py - Training script for SQuAD Q&A with encoder-decoder architecture

Architecture:
- Encoder: Processes context passage (bidirectional attention)
- Decoder: Generates answer while attending to encoder via cross-attention
- Loss: Only computed on answer tokens (question tokens masked)
"""
import argparse
import json
import math
import os
import time
import datetime

import torch
import torch.nn.functional as F
from datasets import load_dataset

from architecture.transformer import Transformer
from architecture.config import ModelConfig
from architecture.encoder import BidirectionalEncoder
from architecture.tokenizer import get_tokenizer

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

# ------------------------------- args ----------------------------------------
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="model_squad")
    ap.add_argument("--model_size", type=str, choices=["toy", "small", "medium", "large"], default="toy")
    # training
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_context_len", type=int, default=512)
    ap.add_argument("--max_qa_len", type=int, default=128)
    ap.add_argument("--max_iters", type=int, default=10000)
    ap.add_argument("--log_interval", type=int, default=10)
    ap.add_argument("--eval_interval", type=int, default=200)
    ap.add_argument("--eval_iters", type=int, default=20)
    # save + sample
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--sample_every", type=int, default=500)
    # optim
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--warmup_iters", type=int, default=200)
    ap.add_argument("--min_lr", type=float, default=3e-5)
    # system
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--dtype", type=str, choices=["float32", "bfloat16", "float16"], default="bfloat16")
    # checkpoint
    ap.add_argument("--resume", action="store_true", default=False)
    ap.add_argument("--checkpoint_path", type=str, default=None)
    # pretrained decoder
    ap.add_argument("--pretrained_decoder_path", type=str, default=None, help="Path to pretrained decoder weights (e.g., from TinyStories)")
    ap.add_argument("--decoder_lr", type=float, default=None, help="Learning rate for pretrained decoder layers (if None, uses --lr)")
    ap.add_argument("--new_layers_lr", type=float, default=None, help="Learning rate for encoder and cross-attention (if None, uses --lr)")
    # tensorboard
    ap.add_argument("--use_tensorboard", action="store_true", default=False)
    ap.add_argument("--log_dir", type=str, default="runs_squad")
    # special tokens
    ap.add_argument("--question_token", type=str, default="<Q>")
    ap.add_argument("--sep_token", type=str, default="<SEP>")
    # encoder compression
    ap.add_argument("--use_lsi_compression", action="store_true", default=False, help="Use LSI cross-attention for encoder compression instead of SVD")
    ap.add_argument("--num_compression_slots", type=int, default=64, help="Number of latent slots for LSI compression")
    return ap.parse_args()

# ------------------------------ pretrained loading --------------------------
def load_pretrained_decoder(decoder, pretrained_path, device):
    """
    Load pretrained decoder weights, handling missing cross-attention layers.
    
    Loads:
    - Token embeddings
    - Self-attention layers (AttentionBlock)
    - MLP layers
    
    Skips (trains from scratch):
    - Cross-attention layers (CrossAttentionLayer)
    - Output projection if vocab size changed
    """
    print(f"Loading pretrained decoder from {pretrained_path}...")
    checkpoint = torch.load(pretrained_path, map_location=device)
    
    # Extract decoder state dict (handle different checkpoint formats)
    if 'model' in checkpoint:
        pretrained_state = checkpoint['model']
    elif 'decoder' in checkpoint:
        pretrained_state = checkpoint['decoder']
    else:
        pretrained_state = checkpoint
    
    # Get current model state
    model_state = decoder.state_dict()
    
    # Track loaded and skipped parameters
    loaded_keys = []
    skipped_keys = []
    
    # Load compatible weights
    for name, param in pretrained_state.items():
        if name in model_state:
            # Check if shapes match
            if param.shape == model_state[name].shape:
                model_state[name] = param
                loaded_keys.append(name)
            else:
                skipped_keys.append(f"{name} (shape mismatch: {param.shape} vs {model_state[name].shape})")
        else:
            # Key doesn't exist in new model (e.g., old checkpoint didn't have cross-attention)
            skipped_keys.append(f"{name} (not in current model)")
    
    # Check for new parameters not in pretrained checkpoint
    new_keys = []
    for name in model_state.keys():
        if name not in pretrained_state:
            new_keys.append(name)
    
    # Load the state dict
    decoder.load_state_dict(model_state)
    
    print(f"✓ Loaded {len(loaded_keys)} parameter tensors from pretrained decoder")
    print(f"✗ Skipped {len(skipped_keys)} incompatible parameters")
    print(f"✓ Initialized {len(new_keys)} new parameters randomly (cross-attention layers)")
    
    if skipped_keys:
        print("\nSkipped parameters:")
        for key in skipped_keys[:5]:  # Show first 5
            print(f"  - {key}")
        if len(skipped_keys) > 5:
            print(f"  ... and {len(skipped_keys) - 5} more")
    
    if new_keys:
        print("\nNew parameters (training from scratch):")
        cross_attn_count = sum(1 for k in new_keys if 'cross_attn' in k)
        other_count = len(new_keys) - cross_attn_count
        print(f"  - Cross-attention layers: {cross_attn_count} parameters")
        if other_count > 0:
            print(f"  - Other new parameters: {other_count}")
    
    return loaded_keys, new_keys

# ------------------------------ config --------------------------------------
def build_config(name: str, vocab_size: int, use_lsi_compression: bool = False, num_compression_slots: int = 64) -> tuple[object, ModelConfig]:
    """Build encoder and decoder configs based on size"""
    from architecture.encoder import EncoderConfig
    
    if name == "large":
        encoder_config = EncoderConfig(
            vocab_size=vocab_size,
            hidden_size=2048,
            num_hidden_layers=12,
            head_dim=64,
            num_attention_heads=32,
            num_key_value_heads=8,
            num_experts=16,
            experts_per_token=2,
            intermediate_size=2048,
            max_position_embeddings=4096,
            use_lsi_compression=use_lsi_compression,
            num_compression_slots=num_compression_slots,
        )
        decoder_config = ModelConfig(
            vocab_size=vocab_size,
            hidden_size=2048,
            num_hidden_layers=12,
            head_dim=64,
            num_attention_heads=32,
            num_key_value_heads=8,
            num_experts=16,
            experts_per_token=2,
            intermediate_size=2048,
            sliding_window=128,
            initial_context_length=2048,
            use_lsi_cross_attention=False,
            use_context_embedding=True,
            use_encoder_decoder_cross_attention=True,  # Enable cross-attention
        )
    elif name == "medium":
        encoder_config = EncoderConfig(
            vocab_size=vocab_size,
            hidden_size=1024,
            num_hidden_layers=8,
            head_dim=64,
            num_attention_heads=16,
            num_key_value_heads=4,
            num_experts=8,
            experts_per_token=2,
            intermediate_size=1024,
            max_position_embeddings=2048,
            use_lsi_compression=use_lsi_compression,
            num_compression_slots=num_compression_slots,
        )
        decoder_config = ModelConfig(
            vocab_size=vocab_size,
            hidden_size=1024,
            num_hidden_layers=8,
            head_dim=64,
            num_attention_heads=16,
            num_key_value_heads=4,
            num_experts=8,
            experts_per_token=2,
            intermediate_size=1024,
            sliding_window=128,
            initial_context_length=1024,
            use_lsi_cross_attention=False,
            use_context_embedding=True,
            use_encoder_decoder_cross_attention=True,
        )
    elif name == "small":
        encoder_config = EncoderConfig(
            vocab_size=vocab_size,
            hidden_size=512,
            num_hidden_layers=6,
            head_dim=64,
            num_attention_heads=8,
            num_key_value_heads=2,
            num_experts=4,
            experts_per_token=2,
            intermediate_size=512,
            max_position_embeddings=1024,
            use_lsi_compression=use_lsi_compression,
            num_compression_slots=num_compression_slots,
        )
        decoder_config = ModelConfig(
            vocab_size=vocab_size,
            hidden_size=512,
            num_hidden_layers=6,
            head_dim=64,
            num_attention_heads=8,
            num_key_value_heads=2,
            num_experts=4,
            experts_per_token=2,
            intermediate_size=512,
            sliding_window=128,
            initial_context_length=512,
            use_lsi_cross_attention=False,
            use_context_embedding=True,
            use_encoder_decoder_cross_attention=True,
        )
    else:  # toy
        encoder_config = EncoderConfig(
            vocab_size=vocab_size,
            hidden_size=256,
            num_hidden_layers=4,
            head_dim=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_experts=4,
            experts_per_token=2,
            intermediate_size=256,
            max_position_embeddings=512,
            use_lsi_compression=use_lsi_compression,
            num_compression_slots=num_compression_slots,
        )
        decoder_config = ModelConfig(
            vocab_size=vocab_size,
            hidden_size=256,
            num_hidden_layers=4,
            head_dim=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_experts=4,
            experts_per_token=2,
            intermediate_size=256,
            sliding_window=64,
            initial_context_length=256,
            use_lsi_cross_attention=False,
            use_context_embedding=True,
            use_encoder_decoder_cross_attention=True,
        )
    
    return encoder_config, decoder_config

# ------------------------------ data ----------------------------------------
def format_squad_example(example, tokenizer, max_context_len, max_answer_len, q_token, sep_token):
    """
    Format a single SQuAD example into encoder input and decoder input/target.
    
    New Architecture:
    - Encoder: <C> Context <SEP> Question (bidirectional, question-aware context encoding)
    - Decoder: Question <A> Answer (generates answer with self+cross attention)
    - Loss: Only computed on answer tokens (after <A>)
    
    Returns:
        encoder_tokens: <C> + Context + <SEP> + Question (for encoder)
        decoder_tokens: Question + <A> + Answer (for decoder input during training)
        target_tokens: decoder_tokens shifted by 1 (for loss computation)
        a_position: Position of <A> in decoder_tokens (for loss masking)
    """
    context = example['context']
    question = example['question']
    
    # Get answer (SQuAD has multiple answers, take the first)
    if len(example['answers']['text']) > 0:
        answer = example['answers']['text'][0]
    else:
        return None  # Skip examples without answers
    
    # Tokenize
    context_tokens = tokenizer.encode(context)
    question_tokens = tokenizer.encode(question)
    answer_tokens = tokenizer.encode(answer)
    
    # Special tokens
    c_marker = tokenizer.encode("<C>")
    sep_marker = tokenizer.encode(sep_token)
    a_marker = tokenizer.encode("<A>")
    
    # Build encoder input: <C> Context <SEP> Question
    # Reserve space for question, separator, and context marker
    max_context_space = max_context_len - len(question_tokens) - len(sep_marker) - len(c_marker)
    if max_context_space < 50:  # Need at least some context
        return None
    
    if len(context_tokens) > max_context_space:
        context_tokens = context_tokens[:max_context_space]
    
    encoder_tokens = c_marker + context_tokens + sep_marker + question_tokens
    
    # Build decoder input: Question <A> Answer
    # Reserve space for question, <A> marker, and answer
    max_answer_space = max_answer_len - len(question_tokens) - len(a_marker)
    if max_answer_space < 10:  # Need space for some answer
        return None
    
    if len(answer_tokens) > max_answer_space:
        answer_tokens = answer_tokens[:max_answer_space]
    
    decoder_tokens = question_tokens + a_marker + answer_tokens
    
    # Target: decoder_tokens shifted by 1 (next token prediction)
    target_tokens = decoder_tokens[1:] + [tokenizer.encode("<|endoftext|>")[0] if hasattr(tokenizer, 'encode') else 0]
    
    # Position of <A> marker in decoder_tokens (for loss masking)
    a_position = len(question_tokens)  # <A> is right after question
    
    return {
        'encoder_tokens': encoder_tokens,
        'decoder_tokens': decoder_tokens,
        'target_tokens': target_tokens,
        'a_position': a_position,
        'question': question,
        'answer': answer,
    }

def create_squad_dataloaders(tokenizer, max_context_len, max_answer_len, batch_size, q_token, sep_token):
    """Create SQuAD dataloaders"""
    print("Loading SQuAD dataset...")
    dataset = load_dataset("squad")
    
    print("Formatting examples...")
    train_examples = []
    val_examples = []
    
    # Process training set
    for example in dataset['train']:
        formatted = format_squad_example(
            example, tokenizer, max_context_len, max_answer_len, q_token, sep_token
        )
        if formatted is not None:
            train_examples.append(formatted)
    
    # Process validation set
    for example in dataset['validation']:
        formatted = format_squad_example(
            example, tokenizer, max_context_len, max_answer_len, q_token, sep_token
        )
        if formatted is not None:
            val_examples.append(formatted)
    
    print(f"Training examples: {len(train_examples)}")
    print(f"Validation examples: {len(val_examples)}")
    
    return train_examples, val_examples

def get_batch(examples, batch_size, device):
    """Get a random batch of examples"""
    import random
    batch = random.sample(examples, min(batch_size, len(examples)))
    
    # Find max lengths in batch
    max_encoder = max(len(ex['encoder_tokens']) for ex in batch)
    max_decoder = max(len(ex['decoder_tokens']) for ex in batch)
    max_target = max(len(ex['target_tokens']) for ex in batch)
    
    # Prepare batch tensors
    encoder_batch = []
    decoder_batch = []
    target_batch = []
    a_position_batch = []
    
    for ex in batch:
        # Pad encoder (context + question)
        encoder = ex['encoder_tokens'] + [0] * (max_encoder - len(ex['encoder_tokens']))
        encoder_batch.append(encoder)
        
        # Pad decoder (Question <A> Answer)
        decoder = ex['decoder_tokens'] + [0] * (max_decoder - len(ex['decoder_tokens']))
        decoder_batch.append(decoder)
        
        # Pad targets
        target = ex['target_tokens'] + [-100] * (max_target - len(ex['target_tokens']))
        target_batch.append(target)
        
        # Track position of <A> for loss masking
        a_position_batch.append(ex['a_position'])
    
    return (
        torch.tensor(encoder_batch, dtype=torch.long, device=device),
        torch.tensor(decoder_batch, dtype=torch.long, device=device),
        torch.tensor(target_batch, dtype=torch.long, device=device),
        a_position_batch,  # List of integers (not tensor)
    )

# ------------------------------ training ------------------------------------
def get_lr(it: int, warmup_iters: int, max_iters: int, lr: float, min_lr: float) -> float:
    """Cosine learning rate schedule with warmup"""
    if it < warmup_iters:
        return lr * it / warmup_iters
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)

@torch.no_grad()
def evaluate(encoder, decoder, val_examples, eval_iters, device, dtype_ctx, batch_size):
    """Evaluate the model on validation data"""
    encoder.eval()
    decoder.eval()
    
    losses = []
    
    for _ in range(eval_iters):
        encoder_batch, decoder_batch, target_batch, a_position_batch = get_batch(val_examples, batch_size, device)
        
        batch_loss = 0.0
        for i in range(encoder_batch.shape[0]):
            encoder_tokens = encoder_batch[i]
            decoder_tokens = decoder_batch[i]
            targets = target_batch[i]
            a_position = a_position_batch[i]
            
            # Remove padding
            encoder_tokens = encoder_tokens[encoder_tokens != 0]
            decoder_mask = decoder_tokens != 0
            decoder_tokens = decoder_tokens[decoder_mask]
            targets = targets[targets != -100]
            
            with torch.amp.autocast(device_type=device.type, dtype=dtype_ctx):
                # Encode context + question
                encoder_k, encoder_v = encoder(encoder_tokens, return_encoder_kv=True)
                
                # Decode answer with cross-attention
                decoder.reset_context()
                logits = decoder(decoder_tokens, encoder_k=encoder_k, encoder_v=encoder_v)
                
                # Loss masking: only compute loss on answer tokens (after <A>)
                # logits[:-1] predicts targets (shifted by 1)
                seq_len = len(logits) - 1
                loss_mask = torch.zeros(seq_len, device=device, dtype=torch.float)
                # Safeguard: if a_position is invalid, compute loss on all tokens
                if a_position is not None and 0 <= a_position < seq_len:
                    loss_mask[a_position:] = 1.0  # Only compute loss from <A> position onwards
                else:
                    loss_mask[:] = 1.0  # Fallback: loss on all tokens
                
                # Compute cross entropy per token
                ce_loss = F.cross_entropy(
                    logits[:-1].view(-1, logits.size(-1)),
                    targets[:seq_len].view(-1),
                    reduction='none'
                )
                
                # Apply mask and compute mean over answer tokens only
                masked_loss = ce_loss * loss_mask
                loss = masked_loss.sum() / (loss_mask.sum() + 1e-8)
            
            batch_loss += loss.item()
        
        losses.append(batch_loss / encoder_batch.shape[0])
    
    encoder.train()
    decoder.train()
    
    return sum(losses) / len(losses) if losses else float('inf')

@torch.no_grad()
def generate_sample(encoder, decoder, tokenizer, context, question, max_tokens, device, q_token, sep_token):
    """Generate answer for a given context and question"""
    encoder.eval()
    decoder.eval()
    
    # Tokenize and build encoder input: <C> context <SEP> question
    c_marker = tokenizer.encode("<C>")
    context_tokens = tokenizer.encode(context)
    question_tokens = tokenizer.encode(question)
    sep_marker = tokenizer.encode(sep_token)
    
    encoder_tokens = c_marker + context_tokens + sep_marker + question_tokens
    encoder_tokens = torch.tensor(encoder_tokens, dtype=torch.long, device=device)
    
    # Encode context + question
    encoder_k, encoder_v = encoder(encoder_tokens, return_encoder_kv=True)
    
    # Start decoder with Question <A> (same as training format)
    a_marker = tokenizer.encode("<A>")
    tokens = torch.tensor(question_tokens + a_marker, dtype=torch.long, device=device)
    
    # Generate answer
    decoder.reset_context()
    generated = []
    
    for _ in range(max_tokens):
        logits = decoder(tokens, encoder_k=encoder_k, encoder_v=encoder_v)
        next_token = torch.argmax(logits[-1], dim=-1).item()
        generated.append(next_token)
        tokens = torch.cat([tokens, torch.tensor([next_token], device=device)])
        
        # Stop at end of sequence
        if next_token == 0:
            break
    
    result = tokenizer.decode(generated)
    
    encoder.train()
    decoder.train()
    
    return result

# ------------------------------ main ----------------------------------------
def main():
    args = get_args()
    
    # Setup
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    
    # Determine dtype
    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    dtype_ctx = dtype_map[args.dtype]
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Setup TensorBoard
    writer = None
    if args.use_tensorboard and TENSORBOARD_AVAILABLE:
        log_dir = os.path.join(args.log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard logging to: {log_dir}")
    
    # Get tokenizer
    print("Loading tokenizer...")
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.n_vocab
    
    # Create configs
    print(f"Building {args.model_size} model configs...")
    encoder_config, decoder_config = build_config(
        args.model_size, 
        vocab_size, 
        use_lsi_compression=args.use_lsi_compression,
        num_compression_slots=args.num_compression_slots
    )
    
    if args.use_lsi_compression:
        print(f"Using LSI compression with {args.num_compression_slots} latent slots")
    else:
        print("Using SVD compression")
    
    # Save configs
    with open(os.path.join(args.out_dir, "encoder_config.json"), "w") as f:
        json.dump({k: v for k, v in encoder_config.__dict__.items() if not k.startswith('_')}, f, indent=2)
    with open(os.path.join(args.out_dir, "decoder_config.json"), "w") as f:
        json.dump({k: v for k, v in decoder_config.__dict__.items() if not k.startswith('_')}, f, indent=2)
    
    # Create models
    print("Initializing encoder and decoder...")
    encoder = BidirectionalEncoder(encoder_config, device=device)
    decoder = Transformer(decoder_config, device=device)
    
    encoder.train()
    decoder.train()
    
    # Load pretrained decoder if specified
    pretrained_decoder_keys = []
    new_decoder_keys = []
    if args.pretrained_decoder_path:
        pretrained_decoder_keys, new_decoder_keys = load_pretrained_decoder(
            decoder, args.pretrained_decoder_path, device
        )
    
    # Count parameters
    encoder_params = sum(p.numel() for p in encoder.parameters())
    decoder_params = sum(p.numel() for p in decoder.parameters())
    print(f"Encoder parameters: {encoder_params:,}")
    print(f"Decoder parameters: {decoder_params:,}")
    print(f"Total parameters: {encoder_params + decoder_params:,}")
    
    # Load SQuAD data
    train_examples, val_examples = create_squad_dataloaders(
        tokenizer, args.max_context_len, args.max_qa_len, args.batch_size,
        args.question_token, args.sep_token
    )
    
    # Setup optimizer with differential learning rates
    print("Setting up optimizer...")
    
    # Determine learning rates
    decoder_lr = args.decoder_lr if args.decoder_lr is not None else args.lr
    new_layers_lr = args.new_layers_lr if args.new_layers_lr is not None else args.lr
    
    # Separate parameter groups if using pretrained decoder
    if args.pretrained_decoder_path and (decoder_lr != new_layers_lr):
        print(f"Using differential learning rates:")
        print(f"  - Pretrained decoder layers: {decoder_lr:.6f}")
        print(f"  - Encoder + Cross-attention: {new_layers_lr:.6f}")
        
        # Get pretrained decoder parameters
        pretrained_params = []
        new_params = []
        
        for name, param in decoder.named_parameters():
            # Check if this parameter was loaded from pretrained
            if any(name == key for key in pretrained_decoder_keys):
                pretrained_params.append(param)
            else:
                new_params.append(param)
        
        # Add all encoder parameters to new_params
        new_params.extend(encoder.parameters())
        
        optimizer = torch.optim.AdamW([
            {'params': pretrained_params, 'lr': decoder_lr},
            {'params': new_params, 'lr': new_layers_lr},
        ], betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
        
        print(f"  - Pretrained parameters: {sum(p.numel() for p in pretrained_params):,}")
        print(f"  - New parameters: {sum(p.numel() for p in new_params):,}")
    else:
        # Single learning rate for all parameters
        all_params = list(encoder.parameters()) + list(decoder.parameters())
        optimizer = torch.optim.AdamW(
            all_params,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay
        )
        print(f"Using single learning rate: {args.lr:.6f}")
    
    # Load checkpoint if resuming
    start_iter = 0
    if args.resume and args.checkpoint_path and os.path.exists(args.checkpoint_path):
        print(f"Loading checkpoint from {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_iter = checkpoint.get('iter', 0)
        print(f"Resuming from iteration {start_iter}")
    
    # Training loop
    print("Starting training...")
    print(f"Max iterations: {args.max_iters}")
    print("-" * 80)
    
    iter_num = start_iter
    running_loss = 0.0
    log_loss_count = 0
    best_val_loss = float('inf')
    
    t0 = time.time()
    
    while iter_num < args.max_iters:
        # Get batch
        encoder_batch, decoder_batch, target_batch, a_position_batch = get_batch(train_examples, args.batch_size, device)
        
        # Update learning rate with cosine schedule
        if args.pretrained_decoder_path and (decoder_lr != new_layers_lr):
            # Different schedules for pretrained vs new layers
            pretrained_lr = get_lr(iter_num, args.warmup_iters, args.max_iters, decoder_lr, args.min_lr)
            new_lr = get_lr(iter_num, args.warmup_iters, args.max_iters, new_layers_lr, args.min_lr)
            optimizer.param_groups[0]['lr'] = pretrained_lr  # Pretrained decoder
            optimizer.param_groups[1]['lr'] = new_lr  # Encoder + cross-attention
            lr = new_lr  # For logging, use higher LR
        else:
            # Single learning rate
            lr = get_lr(iter_num, args.warmup_iters, args.max_iters, args.lr, args.min_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        # Forward pass
        total_loss = 0.0
        for i in range(encoder_batch.shape[0]):
            encoder_tokens = encoder_batch[i]
            decoder_tokens = decoder_batch[i]
            targets = target_batch[i]
            a_position = a_position_batch[i]
            
            # Remove padding
            encoder_tokens = encoder_tokens[encoder_tokens != 0]
            decoder_mask = decoder_tokens != 0
            decoder_tokens = decoder_tokens[decoder_mask]
            targets = targets[targets != -100]
            
            with torch.amp.autocast(device_type=device.type, dtype=dtype_ctx):
                # Encode context + question
                encoder_k, encoder_v = encoder(encoder_tokens, return_encoder_kv=True)
                
                # Decode answer with cross-attention
                decoder.reset_context()
                logits = decoder(decoder_tokens, encoder_k=encoder_k, encoder_v=encoder_v)
                
                # Loss masking: only compute loss on answer tokens (after <A>)
                seq_len = len(logits) - 1
                loss_mask = torch.zeros(seq_len, device=device, dtype=torch.float)
                # Safeguard: if a_position is invalid, compute loss on all tokens
                if a_position is not None and 0 <= a_position < seq_len:
                    loss_mask[a_position:] = 1.0  # Only compute loss from <A> position onwards
                else:
                    loss_mask[:] = 1.0  # Fallback: loss on all tokens
                
                # Compute cross entropy per token
                ce_loss = F.cross_entropy(
                    logits[:-1].view(-1, logits.size(-1)),
                    targets[:seq_len].view(-1),
                    reduction='none'
                )
                
                # Apply mask and compute mean over answer tokens only
                masked_loss = ce_loss * loss_mask
                loss = masked_loss.sum() / (loss_mask.sum() + 1e-8)
            
            total_loss += loss
        
        # Average loss over batch
        loss = total_loss / encoder_batch.shape[0]
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Gradient clipping
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(all_params, args.grad_clip)
        
        optimizer.step()
        
        # Logging
        running_loss += loss.item()
        log_loss_count += 1
        
        if (iter_num + 1) % args.log_interval == 0:
            avg_loss = running_loss / log_loss_count
            t1 = time.time()
            dt = t1 - t0
            print(f"iter {iter_num + 1:5d} | loss {avg_loss:.4f} | lr {lr:.6f} | {dt*1000:.2f}ms")
            
            if writer:
                writer.add_scalar('Loss/train', avg_loss, iter_num + 1)
                writer.add_scalar('Learning_rate', lr, iter_num + 1)
            
            running_loss = 0.0
            log_loss_count = 0
            t0 = time.time()
        
        # Evaluation
        if (iter_num + 1) % args.eval_interval == 0:
            val_loss = evaluate(encoder, decoder, val_examples, args.eval_iters, device, dtype_ctx, args.batch_size)
            print(f"iter {iter_num + 1:5d} | val_loss {val_loss:.4f}")
            
            if writer:
                writer.add_scalar('Loss/val', val_loss, iter_num + 1)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(args.out_dir, "best_model.pt")
                torch.save({
                    'encoder': encoder.state_dict(),
                    'decoder': decoder.state_dict(),
                    'iter': iter_num + 1,
                    'val_loss': val_loss,
                }, best_path)
                print(f"Saved best model (val_loss={val_loss:.4f})")
        
        # Sample generation
        if (iter_num + 1) % args.sample_every == 0:
            # Take a random validation example
            import random
            sample_ex = random.choice(val_examples)
            
            # Decode tokens for display
            encoder_text = tokenizer.decode(sample_ex['encoder_tokens'])
            gold_answer = sample_ex.get('answer', tokenizer.decode(sample_ex['target_tokens']))
            
            # Generate answer
            generated = generate_sample(
                encoder, decoder, tokenizer,
                tokenizer.decode(sample_ex['encoder_tokens'][:256]),  # Use partial context
                sample_ex.get('question', ''),
                max_tokens=50, device=device,
                q_token=args.question_token, sep_token=args.sep_token
            )
            
            print(f"\n{'='*80}\nSample at iter {iter_num + 1}:")
            print(f"Question: {sample_ex.get('question', 'N/A')}")
            print(f"Gold Answer: {gold_answer}")
            print(f"Generated: {generated}")
            print(f"Context: {encoder_text[:200]}...")
            print(f"{'='*80}\n")
        
        # Save checkpoint
        if (iter_num + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.out_dir, f"checkpoint_{iter_num + 1}.pt")
            torch.save({
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iter': iter_num + 1,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
        
        iter_num += 1
    
    # Save final model
    final_path = os.path.join(args.out_dir, "final_model.pt")
    torch.save({
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
    }, final_path)
    print(f"Training complete! Final model saved to {final_path}")
    
    if writer:
        writer.close()


if __name__ == "__main__":
    main()
