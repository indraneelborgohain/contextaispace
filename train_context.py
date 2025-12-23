#!/usr/bin/env python3
"""
train_context.py - Training script for context-aware gptoss_context.py model
Key feature: Resets context vector at document boundaries
"""
import argparse
import json
import math
import os
import time
import datetime

import torch
import torch.nn.functional as F

from architecture.gptoss_context import Transformer, ModelConfig
from architecture.tokenizer import get_tokenizer
from training.trainer import clear_gpu_memory
from training.data_loader_context import create_context_dataloaders

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

# ------------------------------- args ----------------------------------------
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="model_context")
    ap.add_argument("--model_size", type=str, choices=["toy", "medium", "large"], default="toy")
    # training
    ap.add_argument("--batch_size", type=int, default=5)
    ap.add_argument("--block_size", type=int, default=512)
    ap.add_argument("--max_iters", type=int, default=5000)
    ap.add_argument("--log_interval", type=int, default=10)
    ap.add_argument("--eval_interval", type=int, default=150)
    ap.add_argument("--eval_iters", type=int, default=5)
    # save + sample
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--sample_every", type=int, default=250)
    ap.add_argument("--sample_tokens", type=int, default=100)
    ap.add_argument("--top_k", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    # optim
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--warmup_iters", type=int, default=100)
    ap.add_argument("--min_lr", type=float, default=3e-5)
    # system
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", type=str, default="cuda:0", help="Device to use (cuda:0, cpu, etc.)")
    ap.add_argument("--dtype", type=str, choices=["float32", "bfloat16", "float16"], default="bfloat16")
    # checkpoint
    ap.add_argument("--resume", action="store_true", default=False)
    ap.add_argument("--checkpoint_path", type=str, default=None, help="Path to checkpoint to resume from")
    # tensorboard
    ap.add_argument("--use_tensorboard", action="store_true", default=False, help="Enable TensorBoard logging")
    ap.add_argument("--log_dir", type=str, default="runs_context", help="TensorBoard log directory")
    return ap.parse_args()

# ------------------------------ helpers --------------------------------------
def build_config(name: str, vocab_size: int) -> ModelConfig:
    """Build model configuration based on size"""
    if name == "large":
        return ModelConfig(
            vocab_size=vocab_size,
            hidden_size=2048,
            num_hidden_layers=24,
            head_dim=64,
            num_attention_heads=32,
            num_key_value_heads=8,
            num_experts=16,
            experts_per_token=2,
            intermediate_size=2048,
            sliding_window=128,
            initial_context_length=4096,
        )
    elif name == "medium":
        return ModelConfig(
            vocab_size=vocab_size,
            hidden_size=1024,
            num_hidden_layers=12,
            head_dim=64,
            num_attention_heads=16,
            num_key_value_heads=4,
            num_experts=8,
            experts_per_token=2,
            intermediate_size=1024,
            sliding_window=128,
            initial_context_length=2048,
        )
    else:  # toy
        return ModelConfig(
            vocab_size=vocab_size,
            hidden_size=512,
            num_hidden_layers=6,
            head_dim=64,
            num_attention_heads=8,
            num_key_value_heads=2,
            num_experts=4,
            experts_per_token=2,
            intermediate_size=512,
            sliding_window=64,
            initial_context_length=1024,
        )


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
def evaluate(model, val_loader, eval_iters, device, dtype_ctx):
    """
    Evaluate the model on validation data.
    IMPORTANT: Resets context at the start of each document.
    """
    model.eval()
    losses = []
    
    for i, batch in enumerate(val_loader):
        if i >= eval_iters:
            break
        
        inputs, targets, doc_starts, lengths = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Process each sequence in the batch
        batch_losses = []
        for j in range(inputs.shape[0]):
            # Reset context if this is a new document
            if doc_starts[j]:
                model.reset_context()
            
            seq_input = inputs[j]
            seq_target = targets[j]
            seq_len = lengths[j].item()
            
            # Only use actual tokens (not padding)
            seq_input = seq_input[:seq_len]
            seq_target = seq_target[:seq_len]
            
            with torch.amp.autocast(device_type=device.type, dtype=dtype_ctx):
                logits = model(seq_input, update_context=False)  # Don't update context during eval
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    seq_target.view(-1),
                    ignore_index=-100
                )
            
            batch_losses.append(loss.item())
        
        if batch_losses:
            losses.append(sum(batch_losses) / len(batch_losses))
    
    model.train()
    clear_gpu_memory()
    
    return sum(losses) / len(losses) if losses else float('inf')


@torch.no_grad()
def generate_sample(model, tokenizer, prompt, max_tokens, temperature, top_k, device):
    """Generate a sample from the model (resets context before generation)"""
    model.eval()
    model.reset_context()  # Start fresh for generation
    
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device)
    
    for _ in range(max_tokens):
        logits = model(tokens[-512:], update_context=True)  # Use last 512 tokens as context
        logits = logits[-1] / temperature
        
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[-1]] = -float('inf')
        
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token])
    
    result = tokenizer.decode(tokens.tolist())
    model.train()
    return result


# ------------------------------ main ----------------------------------------
def main():
    args = get_args()
    
    # Setup
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    
    # Determine dtype
    if args.dtype == "float32":
        dtype = torch.float32
        dtype_ctx = torch.float32
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
        dtype_ctx = torch.bfloat16
    else:  # float16
        dtype = torch.float16
        dtype_ctx = torch.float16
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Setup TensorBoard
    writer = None
    if args.use_tensorboard and TENSORBOARD_AVAILABLE:
        log_dir = os.path.join(args.log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard logging to: {log_dir}")
        print(f"Run: tensorboard --logdir={args.log_dir}")
    
    # Get tokenizer
    print("Loading tokenizer...")
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.n_vocab
    
    # Create model config
    print(f"Building {args.model_size} model config...")
    config = build_config(args.model_size, vocab_size)
    
    # Save config
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        config_dict = {
            k: v for k, v in config.__dict__.items()
            if not k.startswith('_') and not callable(v)
        }
        json.dump(config_dict, f, indent=2)
    
    # Create model
    print("Initializing model...")
    model = Transformer(config, device=device)
    model.train()
    
    # Load checkpoint if resuming
    start_iter = 0
    if args.resume and args.checkpoint_path and os.path.exists(args.checkpoint_path):
        print(f"Loading checkpoint from {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        start_iter = checkpoint.get('iter', 0)
        print(f"Resuming from iteration {start_iter}")
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Create dataloaders with document boundaries
    print("Creating context-aware dataloaders...")
    train_loader, val_loader = create_context_dataloaders(
        batch_size=args.batch_size,
        max_length=args.block_size,
        num_workers=4,
        shuffle_train=True
    )
    
    # Setup optimizer
    print("Setting up optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay
    )
    
    if args.resume and args.checkpoint_path and os.path.exists(args.checkpoint_path):
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("Loaded optimizer state")
    
    # Training loop
    print("Starting training...")
    print(f"Max iterations: {args.max_iters}")
    print(f"Log interval: {args.log_interval}")
    print(f"Eval interval: {args.eval_interval}")
    print(f"Save interval: {args.save_every}")
    print("-" * 80)
    
    iter_num = start_iter
    train_iter = iter(train_loader)
    running_loss = 0.0
    log_loss_count = 0
    
    t0 = time.time()
    
    while iter_num < args.max_iters:
        # Get batch
        try:
            inputs, targets, doc_starts, lengths = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            inputs, targets, doc_starts, lengths = next(train_iter)
        
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Update learning rate
        lr = get_lr(iter_num, args.warmup_iters, args.max_iters, args.lr, args.min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Forward pass for each sequence in batch
        # CRITICAL: Reset context at document boundaries
        total_loss = 0.0
        valid_samples = 0
        
        for j in range(inputs.shape[0]):
            # Reset context if this is a new document
            if doc_starts[j]:
                model.reset_context()
            
            seq_input = inputs[j]
            seq_target = targets[j]
            seq_len = lengths[j].item()
            
            # Only use actual tokens (not padding)
            seq_input = seq_input[:seq_len]
            seq_target = seq_target[:seq_len]
            
            with torch.amp.autocast(device_type=device.type, dtype=dtype_ctx):
                logits = model(seq_input, update_context=True)  # Update context during training
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    seq_target.view(-1),
                    ignore_index=-100
                )
            
            total_loss += loss
            valid_samples += 1
        
        # Average loss over batch
        if valid_samples > 0:
            loss = total_loss / valid_samples
        else:
            continue  # Skip this batch if no valid samples
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Gradient clipping
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
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
            val_loss = evaluate(model, val_loader, args.eval_iters, device, dtype_ctx)
            print(f"iter {iter_num + 1:5d} | val_loss {val_loss:.4f}")
            
            if writer:
                writer.add_scalar('Loss/val', val_loss, iter_num + 1)
        
        # Sample generation
        if (iter_num + 1) % args.sample_every == 0:
            sample = generate_sample(
                model, tokenizer, "Once upon a time",
                args.sample_tokens, args.temperature, args.top_k, device
            )
            print(f"\n{'='*80}\nSample at iter {iter_num + 1}:\n{sample}\n{'='*80}\n")
            
            if writer:
                writer.add_text('Samples', sample, iter_num + 1)
        
        # Save checkpoint
        if (iter_num + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.out_dir, f"checkpoint_{iter_num + 1}.pt")
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iter': iter_num + 1,
                'config': config.__dict__,
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
        
        iter_num += 1
    
    # Save final model
    final_path = os.path.join(args.out_dir, "final_model.pt")
    torch.save({
        'model': model.state_dict(),
        'config': config.__dict__,
    }, final_path)
    print(f"Training complete! Final model saved to {final_path}")
    
    if writer:
        writer.close()


if __name__ == "__main__":
    main()
