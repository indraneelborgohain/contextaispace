#!/usr/bin/env python3
"""
train.py - Simple training script for gptoss Transformer
Reuses existing training infrastructure with cleaner structure
"""
import argparse
import json
import math
import os
import time
import datetime

import torch
import torch.nn.functional as F

from architecture.gptoss import Transformer, ModelConfig
from architecture.tokenizer import get_tokenizer
from training.trainer import clear_gpu_memory

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

# ------------------------------- args ----------------------------------------
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="model")
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
    ap.add_argument("--log_dir", type=str, default="runs", help="TensorBoard log directory")
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
            initial_context_length=4096,
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
            initial_context_length=2048,
        )

def get_lr(it: int, warmup_iters: int, max_iters: int, lr: float, min_lr: float) -> float:
    """Cosine learning rate schedule with warmup"""
    if it < warmup_iters:
        return lr * it / max(1, warmup_iters)
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)

@torch.no_grad()
def generate_text(model, tokenizer, device, prompt="", n_tokens=100, temperature=0.8, top_k=200):
    """Generate text from the model"""
    model.eval()
    
    # Start with prompt or newline
    if prompt:
        tokens = tokenizer.encode(prompt)
    else:
        try:
            tokens = tokenizer.encode("\n")
        except:
            tokens = [0]
    
    tokens = torch.tensor(tokens, device=device, dtype=torch.long)
    
    for _ in range(n_tokens):
        logits = model(tokens)
        next_logits = logits[-1, :]
        
        # Apply temperature
        if temperature != 1.0:
            next_logits = next_logits / max(1e-6, temperature)
        
        # Apply top-k filtering
        if top_k and top_k > 0:
            v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            next_logits[next_logits < v[-1]] = -float("inf")
        
        probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token])
    
    model.train()
    return tokenizer.decode(tokens.tolist())

@torch.no_grad()
def evaluate(model, val_loader, device, eval_iters, vocab_size):
    """Evaluate model on validation set"""
    model.eval()
    losses = []
    
    for i, (input_batch, target_batch) in enumerate(val_loader):
        if i >= eval_iters:
            break
        
        # Handle batched data
        for j in range(len(input_batch)):
            inp = input_batch[j].to(device, non_blocking=True)
            tgt = target_batch[j].to(device, non_blocking=True)
            
            logits = model(inp)
            loss = F.cross_entropy(logits.view(-1, vocab_size), tgt.view(-1))
            losses.append(loss.item())
    
    model.train()
    clear_gpu_memory()
    
    return sum(losses) / max(1, len(losses))

# -------------------------------- main ---------------------------------------
def main():
    args = get_args()
    
    # Setup
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    print(f"Using device: {device}")
    
    # Initialize TensorBoard if requested
    writer = None
    if args.use_tensorboard and TENSORBOARD_AVAILABLE:
        log_dir = os.path.join(args.log_dir, f"{args.model_size}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard logging to: {log_dir}")
        print(f"Run 'tensorboard --logdir={args.log_dir}' to view")
        
        # Log hyperparameters
        hparams = {
            "model_size": args.model_size,
            "batch_size": args.batch_size,
            "block_size": args.block_size,
            "max_iters": args.max_iters,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "warmup_iters": args.warmup_iters,
        }
        writer.add_text("hyperparameters", str(hparams), 0)
    
    # Tokenizer
    tokenizer = get_tokenizer()
    vocab_size = 201088  # o200k_harmony vocab size
    
    # Model config
    cfg = build_config(args.model_size, vocab_size)
    
    # Create model
    print(f"Creating {args.model_size} model...")
    model = Transformer(cfg, device=device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params/1e6:.2f}M")
    
    # Load existing data loaders
    print("Loading data...")
    try:
        from training.data_loader import train_loader, val_loader
        print(f"Loaded train_loader with {len(train_loader)} batches")
        print(f"Loaded val_loader with {len(val_loader)} batches")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay
    )
    
    # Resume from checkpoint if requested
    iter_num = 0
    best_val_loss = float("inf")
    
    if args.resume:
        checkpoint_path = args.checkpoint_path or os.path.join(args.out_dir, "gptoss.pt")
        if os.path.exists(checkpoint_path):
            print(f"Resuming from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                if "optimizer_state_dict" in checkpoint:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                iter_num = checkpoint.get("iter_num", 0)
                best_val_loss = checkpoint.get("best_val_loss", float("inf"))
                print(f"Resumed from iteration {iter_num}, best val loss: {best_val_loss:.4f}")
            else:
                model.load_state_dict(checkpoint)
                print("Loaded model weights only")
    
    # Training loop
    print(f"\nStarting training for {args.max_iters} iterations...")
    print("=" * 70)
    
    model.train()
    t0 = time.time()
    train_iter = iter(train_loader)
    
    while iter_num < args.max_iters:
        # Learning rate schedule
        lr = get_lr(iter_num, args.warmup_iters, args.max_iters, args.lr, args.min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Get batch
        try:
            input_batch, target_batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            input_batch, target_batch = next(train_iter)
        
        # Forward pass (handle batched data from your DataLoader)
        total_loss = 0.0
        optimizer.zero_grad()
        
        for i in range(len(input_batch)):
            inp = input_batch[i].to(device, non_blocking=True)
            tgt = target_batch[i].to(device, non_blocking=True)
            
            logits = model(inp)
            loss = F.cross_entropy(logits.view(-1, vocab_size), tgt.view(-1))
            loss = loss / len(input_batch)  # Normalize by batch size
            
            total_loss += loss.item()
            loss.backward()
            
            del inp, tgt, logits, loss
        
        # Gradient clipping
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        # Optimizer step
        optimizer.step()
        
        # Clear memory periodically
        if iter_num % 10 == 0:
            clear_gpu_memory()
        
        # Logging
        if iter_num % args.log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()
            print(f"iter {iter_num:06d} | loss {total_loss:.4f} | lr {lr:.6e} | {dt*1000:.1f} ms/it")
            
            if writer is not None:
                writer.add_scalar("train/loss", total_loss, iter_num)
                writer.add_scalar("train/lr", lr, iter_num)
        
        # Evaluation
        if args.eval_interval > 0 and iter_num > 0 and (iter_num % args.eval_interval == 0):
            val_loss = evaluate(model, val_loader, device, args.eval_iters, vocab_size)
            print(f"[eval] iter {iter_num} | val_loss {val_loss:.4f}")
            
            if writer is not None:
                writer.add_scalar("val/loss", val_loss, iter_num)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"✅ New best validation loss: {best_val_loss:.4f}")
                
                # Save best model
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": vars(args),
                }
                torch.save(checkpoint, os.path.join(args.out_dir, "gptoss_best.pt"))
                
                if writer is not None:
                    writer.add_scalar("val/best_loss", best_val_loss, iter_num)
        
        # Sampling
        if args.sample_every > 0 and iter_num > 0 and (iter_num % args.sample_every == 0):
            try:
                txt = generate_text(
                    model, tokenizer, device,
                    prompt="",
                    n_tokens=args.sample_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k
                )
                print("\n--- SAMPLE ---")
                print(txt)
                print("--------------\n")
                
                if writer is not None:
                    writer.add_text("samples/generated_text", txt, iter_num)
            except Exception as e:
                print(f"[sample] Error: {e}")
        
        # Regular checkpointing
        if args.save_every > 0 and iter_num > 0 and (iter_num % args.save_every == 0):
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "iter_num": iter_num,
                "best_val_loss": best_val_loss,
                "config": vars(args),
            }
            torch.save(checkpoint, os.path.join(args.out_dir, f"gptoss_iter{iter_num}.pt"))
            print(f"[ckpt] Saved checkpoint at iteration {iter_num}")
        
        iter_num += 1
    
    # Save final model
    print("\n[train] Training complete!")
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
        "config": vars(args),
    }
    torch.save(checkpoint, os.path.join(args.out_dir, "gptoss.pt"))
    print(f"Saved final model to {args.out_dir}/gptoss.pt")
    if writer is not None:
        writer.close()
        print(f"\nTensorBoard logs saved. View with: tensorboard --logdir={args.log_dir}")

if __name__ == "__main__":
    main()
