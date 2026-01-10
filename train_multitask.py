#!/usr/bin/env python3
"""
train_multitask.py - Multi-task training on TinyStories and SQuAD

Combines training from both datasets to create a more robust encoder-decoder model:
- TinyStories: Story continuation (encoder processes context, decoder generates next chunk)
- SQuAD: Question answering (encoder processes passage, decoder generates answer)

Training modes:
1. Alternating: Alternate between datasets each batch
2. Mixed: Randomly sample from combined dataset pool
3. Sequential: Train on one dataset first, then fine-tune on the other
4. Curriculum: Start with simpler task (stories) then add QA

Architecture:
- Shared encoder: Bidirectional attention for context understanding
- Shared decoder: Generates output while attending to encoder via cross-attention
"""
import argparse
import json
import math
import os
import time
import datetime
import random
from enum import Enum
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn.functional as F
from datasets import load_dataset

from architecture.transformer import Transformer
from architecture.config import ModelConfig
from architecture.encoder import BidirectionalEncoder
from architecture.tokenizer import get_tokenizer
from dataloader.trainer import clear_gpu_memory
from dataloader.data_loader_context import create_context_dataloaders

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None


class TrainingMode(Enum):
    ALTERNATING = "alternating"
    MIXED = "mixed"
    SEQUENTIAL = "sequential"
    CURRICULUM = "curriculum"


# ------------------------------- args ----------------------------------------
def get_args():
    ap = argparse.ArgumentParser(
        description="Multi-task training on TinyStories and SQuAD datasets"
    )
    
    # Output
    ap.add_argument("--out_dir", type=str, default="model_multitask")
    ap.add_argument("--model_size", type=str, choices=["toy", "small", "medium", "large"], default="medium")
    
    # Training mode
    ap.add_argument("--training_mode", type=str, 
                    choices=["alternating", "mixed", "sequential", "curriculum"],
                    default="alternating",
                    help="How to combine the two datasets during training")
    ap.add_argument("--stories_weight", type=float, default=0.5,
                    help="Weight for TinyStories in mixed mode (0-1)")
    ap.add_argument("--curriculum_switch_iter", type=int, default=2500,
                    help="Iteration to switch from stories-only to mixed (curriculum mode)")
    ap.add_argument("--sequential_switch_iter", type=int, default=5000,
                    help="Iteration to switch from first to second dataset (sequential mode)")
    
    # Dataset selection for sequential mode
    ap.add_argument("--sequential_first", type=str, choices=["stories", "squad"], default="stories",
                    help="Which dataset to train on first in sequential mode")
    
    # Training hyperparameters
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--block_size", type=int, default=512,
                    help="Max sequence length for TinyStories")
    ap.add_argument("--encoder_chunk_size", type=int, default=512,
                    help="Size of encoder context chunks (should match block_size)")
    ap.add_argument("--max_context_len", type=int, default=512,
                    help="Max context length for SQuAD")
    ap.add_argument("--max_answer_len", type=int, default=128,
                    help="Max answer length for SQuAD")
    ap.add_argument("--max_iters", type=int, default=10000)
    ap.add_argument("--log_interval", type=int, default=10)
    ap.add_argument("--eval_interval", type=int, default=200)
    ap.add_argument("--eval_iters", type=int, default=10)
    
    # Save + sample
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--sample_every", type=int, default=250)
    ap.add_argument("--sample_tokens", type=int, default=100)
    ap.add_argument("--top_k", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    
    # Optimizer
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--warmup_iters", type=int, default=200)
    ap.add_argument("--min_lr", type=float, default=3e-5)
    
    # Pretrained weights
    ap.add_argument("--pretrained_decoder_path", type=str, default=None,
                    help="Path to pretrained decoder weights")
    ap.add_argument("--pretrained_encoder_decoder_path", type=str, default=None,
                    help="Path to pretrained encoder-decoder checkpoint")
    ap.add_argument("--decoder_lr", type=float, default=None,
                    help="Learning rate for pretrained decoder (if None, uses --lr)")
    ap.add_argument("--new_layers_lr", type=float, default=None,
                    help="Learning rate for encoder and cross-attention (if None, uses --lr)")
    
    # System
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--dtype", type=str, choices=["float32", "bfloat16", "float16"], default="bfloat16")
    
    # Checkpoint
    ap.add_argument("--resume", action="store_true", default=False)
    ap.add_argument("--checkpoint_path", type=str, default=None)
    
    # TensorBoard
    ap.add_argument("--use_tensorboard", action="store_true", default=False)
    ap.add_argument("--log_dir", type=str, default="runs_multitask")
    
    # Special tokens for SQuAD
    ap.add_argument("--question_token", type=str, default="<Q>")
    ap.add_argument("--sep_token", type=str, default="<SEP>")
    
    # Encoder compression
    ap.add_argument("--use_lsi_compression", action="store_true", default=False,
                    help="Use LSI cross-attention for encoder compression")
    ap.add_argument("--num_compression_slots", type=int, default=64,
                    help="Number of latent slots for LSI compression")
    
    # Task-specific loss weighting
    ap.add_argument("--squad_loss_weight", type=float, default=1.0,
                    help="Weight for SQuAD loss")
    ap.add_argument("--stories_loss_weight", type=float, default=1.0,
                    help="Weight for TinyStories loss")
    
    return ap.parse_args()


# ------------------------------ pretrained loading --------------------------
def load_pretrained_decoder(decoder, pretrained_path, device):
    """Load pretrained decoder weights with partial loading support"""
    print(f"Loading pretrained decoder from {pretrained_path}...")
    checkpoint = torch.load(pretrained_path, map_location=device, weights_only=False)
    
    # Extract decoder state dict
    if 'model' in checkpoint:
        pretrained_state = checkpoint['model']
    elif 'decoder' in checkpoint:
        pretrained_state = checkpoint['decoder']
    else:
        pretrained_state = checkpoint
    
    model_state = decoder.state_dict()
    loaded_keys = []
    skipped_keys = []
    partial_keys = []
    
    for name, param in pretrained_state.items():
        if 'cross_attn' in name:
            skipped_keys.append(f"{name} (cross-attention layer)")
            continue
            
        if name in model_state:
            if param.shape == model_state[name].shape:
                model_state[name] = param
                loaded_keys.append(name)
            else:
                current_shape = model_state[name].shape
                pretrained_shape = param.shape
                
                if len(current_shape) == len(pretrained_shape):
                    min_shape = tuple(min(c, p) for c, p in zip(current_shape, pretrained_shape))
                    if all(m > 0 for m in min_shape):
                        slices = tuple(slice(0, m) for m in min_shape)
                        model_state[name][slices] = param[slices]
                        partial_keys.append(f"{name} (partial)")
                    else:
                        skipped_keys.append(f"{name} (shape mismatch)")
                else:
                    skipped_keys.append(f"{name} (dimension mismatch)")
        else:
            skipped_keys.append(f"{name} (not in model)")
    
    new_keys = [name for name in model_state.keys() if name not in pretrained_state]
    decoder.load_state_dict(model_state)
    
    print(f"✓ Loaded {len(loaded_keys)} parameters (exact)")
    print(f"✓ Partially loaded {len(partial_keys)} parameters")
    print(f"✗ Skipped {len(skipped_keys)} parameters")
    print(f"✓ Initialized {len(new_keys)} new parameters")
    
    return loaded_keys, new_keys


def load_pretrained_encoder_decoder(encoder, decoder, checkpoint_path, device):
    """Load both encoder and decoder from a checkpoint"""
    print(f"Loading pretrained encoder-decoder from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    encoder_loaded = 0
    decoder_loaded = 0
    
    if 'encoder' in checkpoint:
        encoder.load_state_dict(checkpoint['encoder'])
        encoder_loaded = sum(p.numel() for p in encoder.parameters())
        print(f"✓ Loaded encoder: {encoder_loaded:,} parameters")
    
    if 'decoder' in checkpoint:
        decoder.load_state_dict(checkpoint['decoder'])
        decoder_loaded = sum(p.numel() for p in decoder.parameters())
        print(f"✓ Loaded decoder: {decoder_loaded:,} parameters")
    
    return checkpoint.get('iter', 0)


# ------------------------------ config --------------------------------------
def build_config(name: str, vocab_size: int, is_encoder: bool = False, 
                 use_lsi_compression: bool = False, num_compression_slots: int = 64):
    """Build encoder or decoder configuration based on size"""
    from architecture.encoder import EncoderConfig
    
    configs = {
        "large": {
            "encoder": EncoderConfig(
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
            ),
            "decoder": ModelConfig(
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
                use_encoder_decoder_cross_attention=True,
            ),
        },
        "medium": {
            "encoder": EncoderConfig(
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
            ),
            "decoder": ModelConfig(
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
            ),
        },
        "small": {
            "encoder": EncoderConfig(
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
            ),
            "decoder": ModelConfig(
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
            ),
        },
        "toy": {
            "encoder": EncoderConfig(
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
            ),
            "decoder": ModelConfig(
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
            ),
        },
    }
    
    config_set = configs[name]
    return config_set["encoder"] if is_encoder else config_set["decoder"]


# ------------------------------ SQuAD data ----------------------------------
def format_squad_example(example, tokenizer, max_context_len, max_answer_len, q_token, sep_token):
    """Format a SQuAD example for encoder-decoder training
    
    New Architecture:
    - Encoder: Context <SEP> Question (question-aware context encoding)
    - Decoder: <A> Answer (generates answer with cross-attention to encoder)
    """
    context = example['context']
    question = example['question']
    
    if len(example['answers']['text']) == 0:
        return None
    answer = example['answers']['text'][0]
    
    context_tokens = tokenizer.encode(context)
    question_tokens = tokenizer.encode(question)
    answer_tokens = tokenizer.encode(answer)
    sep_marker = tokenizer.encode(sep_token)
    a_marker = tokenizer.encode("<A>")
    
    # Build encoder input: Context <SEP> Question
    max_context_space = max_context_len - len(question_tokens) - len(sep_marker)
    if max_context_space < 50:
        return None
    
    if len(context_tokens) > max_context_space:
        context_tokens = context_tokens[:max_context_space]
    
    encoder_tokens = context_tokens + sep_marker + question_tokens
    
    # Build decoder input: <A> Answer
    if len(answer_tokens) > max_answer_len - len(a_marker):
        answer_tokens = answer_tokens[:max_answer_len - len(a_marker)]
    
    decoder_tokens = a_marker + answer_tokens
    target_tokens = answer_tokens + [0]  # Add EOS
    
    return {
        'encoder_tokens': encoder_tokens,
        'decoder_tokens': decoder_tokens,
        'target_tokens': target_tokens,
        'task': 'squad',
        'question': question,
        'answer': answer,
    }


def load_squad_data(tokenizer, max_context_len, max_answer_len, q_token, sep_token):
    """Load and format SQuAD dataset"""
    print("Loading SQuAD dataset...")
    dataset = load_dataset("squad")
    
    train_examples = []
    val_examples = []
    
    for example in dataset['train']:
        formatted = format_squad_example(
            example, tokenizer, max_context_len, max_answer_len, q_token, sep_token
        )
        if formatted is not None:
            train_examples.append(formatted)
    
    for example in dataset['validation']:
        formatted = format_squad_example(
            example, tokenizer, max_context_len, max_answer_len, q_token, sep_token
        )
        if formatted is not None:
            val_examples.append(formatted)
    
    print(f"SQuAD - Train: {len(train_examples)}, Val: {len(val_examples)}")
    return train_examples, val_examples


def get_squad_batch(examples, batch_size, device):
    """Get a random batch from SQuAD examples"""
    batch = random.sample(examples, min(batch_size, len(examples)))
    
    max_encoder = max(len(ex['encoder_tokens']) for ex in batch)
    max_decoder = max(len(ex['decoder_tokens']) for ex in batch)
    
    encoder_batch = []
    decoder_batch = []
    target_batch = []
    
    for ex in batch:
        encoder = ex['encoder_tokens'] + [0] * (max_encoder - len(ex['encoder_tokens']))
        decoder = ex['decoder_tokens'] + [0] * (max_decoder - len(ex['decoder_tokens']))
        target = ex['target_tokens'] + [0] * (max_decoder - len(ex['target_tokens']))
        
        encoder_batch.append(encoder)
        decoder_batch.append(decoder)
        target_batch.append(target)
    
    return (
        torch.tensor(encoder_batch, dtype=torch.long, device=device),
        torch.tensor(decoder_batch, dtype=torch.long, device=device),
        torch.tensor(target_batch, dtype=torch.long, device=device),
    )


# ------------------------------ training helpers ----------------------------
def get_lr(it: int, warmup_iters: int, max_iters: int, lr: float, min_lr: float) -> float:
    """Cosine learning rate schedule with warmup"""
    if it < warmup_iters:
        return lr * it / warmup_iters
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


def train_squad_step(encoder, decoder, batch, device, dtype_ctx, loss_weight=1.0):
    """Single training step on SQuAD data
    
    New Architecture:
    - Encoder: Context <SEP> Question
    - Decoder: <A> Answer (cross-attends to encoder)
    """
    encoder_batch, decoder_batch, target_batch = batch
    
    total_loss = 0.0
    for i in range(encoder_batch.shape[0]):
        encoder_tokens = encoder_batch[i]
        decoder_tokens = decoder_batch[i]
        targets = target_batch[i]
        
        # Remove padding
        encoder_tokens = encoder_tokens[encoder_tokens != 0]
        decoder_mask = decoder_tokens != 0
        decoder_tokens = decoder_tokens[decoder_mask]
        targets = targets[decoder_mask]
        
        with torch.amp.autocast(device_type=device.type, dtype=dtype_ctx):
            encoder_k, encoder_v = encoder(encoder_tokens, return_encoder_kv=True)
            decoder.reset_context()
            logits = decoder(decoder_tokens, encoder_k=encoder_k, encoder_v=encoder_v)
            
            # Loss on answer tokens (logits from <A> and answer predict answer tokens)
            loss = F.cross_entropy(
                logits[:-1].view(-1, logits.size(-1)),
                targets[:len(logits)-1].view(-1),
                ignore_index=0
            )
        
        total_loss += loss
    
    return (total_loss / encoder_batch.shape[0]) * loss_weight


def train_stories_step(encoder, decoder, batch, device, dtype_ctx, loss_weight=1.0):
    """Single training step on TinyStories data"""
    inputs, targets, doc_starts, lengths = batch
    inputs = inputs.to(device)
    targets = targets.to(device)
    
    seq_inputs_list = []
    seq_targets_list = []
    total_loss = 0.0
    valid_samples = 0
    
    for j in range(inputs.shape[0]):
        seq_input = inputs[j]
        seq_target = targets[j]
        seq_len = lengths[j].item()
        
        if seq_len < seq_input.shape[0]:
            seq_input = seq_input[:seq_len]
            seq_target = seq_target[:seq_len]
        
        if doc_starts[j]:
            # Process accumulated sequences from previous document
            if seq_inputs_list:
                full_doc_input = torch.cat(seq_inputs_list, dim=0)
                
                with torch.amp.autocast(device_type=device.type, dtype=dtype_ctx):
                    encoder_k, encoder_v = encoder(full_doc_input, return_encoder_kv=True)
                
                decoder.reset_context()
                for seq_in, seq_tgt in zip(seq_inputs_list, seq_targets_list):
                    with torch.amp.autocast(device_type=device.type, dtype=dtype_ctx):
                        logits = decoder(seq_in, encoder_k=encoder_k, encoder_v=encoder_v)
                        loss = F.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            seq_tgt.view(-1),
                            ignore_index=-100
                        )
                    total_loss += loss
                    valid_samples += 1
            
            seq_inputs_list = []
            seq_targets_list = []
        
        seq_inputs_list.append(seq_input)
        seq_targets_list.append(seq_target)
    
    # Process remaining sequences
    if seq_inputs_list:
        full_doc_input = torch.cat(seq_inputs_list, dim=0)
        
        with torch.amp.autocast(device_type=device.type, dtype=dtype_ctx):
            encoder_k, encoder_v = encoder(full_doc_input, return_encoder_kv=True)
        
        decoder.reset_context()
        for seq_in, seq_tgt in zip(seq_inputs_list, seq_targets_list):
            with torch.amp.autocast(device_type=device.type, dtype=dtype_ctx):
                logits = decoder(seq_in, encoder_k=encoder_k, encoder_v=encoder_v)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    seq_tgt.view(-1),
                    ignore_index=-100
                )
            total_loss += loss
            valid_samples += 1
    
    if valid_samples == 0:
        return None
    
    return (total_loss / valid_samples) * loss_weight


@torch.no_grad()
def evaluate_squad(encoder, decoder, val_examples, eval_iters, device, dtype_ctx, batch_size):
    """Evaluate on SQuAD validation set"""
    encoder.eval()
    decoder.eval()
    
    losses = []
    for _ in range(eval_iters):
        batch = get_squad_batch(val_examples, batch_size, device)
        encoder_batch, decoder_batch, target_batch = batch
        
        batch_loss = 0.0
        for i in range(encoder_batch.shape[0]):
            encoder_tokens = encoder_batch[i]
            decoder_tokens = decoder_batch[i]
            targets = target_batch[i]
            
            encoder_tokens = encoder_tokens[encoder_tokens != 0]
            decoder_mask = decoder_tokens != 0
            decoder_tokens = decoder_tokens[decoder_mask]
            targets = targets[decoder_mask]
            
            with torch.amp.autocast(device_type=device.type, dtype=dtype_ctx):
                encoder_k, encoder_v = encoder(encoder_tokens, return_encoder_kv=True)
                decoder.reset_context()
                logits = decoder(decoder_tokens, encoder_k=encoder_k, encoder_v=encoder_v)
                loss = F.cross_entropy(
                    logits[:-1].view(-1, logits.size(-1)),
                    targets[:len(logits)-1].view(-1),
                    ignore_index=0
                )
            batch_loss += loss.item()
        
        losses.append(batch_loss / encoder_batch.shape[0])
    
    encoder.train()
    decoder.train()
    return sum(losses) / len(losses) if losses else float('inf')


@torch.no_grad()
def evaluate_stories(encoder, decoder, val_loader, eval_iters, device, dtype_ctx):
    """Evaluate on TinyStories validation set"""
    encoder.eval()
    decoder.eval()
    
    losses = []
    for i, batch in enumerate(val_loader):
        if i >= eval_iters:
            break
        
        inputs, targets, doc_starts, lengths = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        seq_inputs_list = []
        seq_targets_list = []
        batch_losses = []
        
        for j in range(inputs.shape[0]):
            seq_input = inputs[j]
            seq_target = targets[j]
            seq_len = lengths[j].item()
            
            if seq_len < seq_input.shape[0]:
                seq_input = seq_input[:seq_len]
                seq_target = seq_target[:seq_len]
            
            if doc_starts[j] and seq_inputs_list:
                full_doc_input = torch.cat(seq_inputs_list, dim=0)
                
                with torch.amp.autocast(device_type=device.type, dtype=dtype_ctx):
                    encoder_k, encoder_v = encoder(full_doc_input, return_encoder_kv=True)
                
                decoder.reset_context()
                for seq_in, seq_tgt in zip(seq_inputs_list, seq_targets_list):
                    with torch.amp.autocast(device_type=device.type, dtype=dtype_ctx):
                        logits = decoder(seq_in, encoder_k=encoder_k, encoder_v=encoder_v)
                        loss = F.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            seq_tgt.view(-1),
                            ignore_index=-100
                        )
                    batch_losses.append(loss.item())
                
                seq_inputs_list = []
                seq_targets_list = []
            
            seq_inputs_list.append(seq_input)
            seq_targets_list.append(seq_target)
        
        # Process remaining
        if seq_inputs_list:
            full_doc_input = torch.cat(seq_inputs_list, dim=0)
            with torch.amp.autocast(device_type=device.type, dtype=dtype_ctx):
                encoder_k, encoder_v = encoder(full_doc_input, return_encoder_kv=True)
            
            decoder.reset_context()
            for seq_in, seq_tgt in zip(seq_inputs_list, seq_targets_list):
                with torch.amp.autocast(device_type=device.type, dtype=dtype_ctx):
                    logits = decoder(seq_in, encoder_k=encoder_k, encoder_v=encoder_v)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        seq_tgt.view(-1),
                        ignore_index=-100
                    )
                batch_losses.append(loss.item())
        
        if batch_losses:
            losses.append(sum(batch_losses) / len(batch_losses))
    
    encoder.train()
    decoder.train()
    clear_gpu_memory()
    return sum(losses) / len(losses) if losses else float('inf')


@torch.no_grad()
def generate_story_sample(encoder, decoder, tokenizer, prompt, max_tokens, temperature, top_k, device):
    """Generate story continuation"""
    encoder.eval()
    decoder.eval()
    
    prompt_tokens = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device)
    encoder_k, encoder_v = encoder(prompt_tokens, return_encoder_kv=True)
    
    decoder.reset_context()
    tokens = prompt_tokens[-50:] if len(prompt_tokens) > 50 else prompt_tokens
    
    for _ in range(max_tokens):
        logits = decoder(tokens, encoder_k=encoder_k, encoder_v=encoder_v)
        logits = logits[-1] / temperature
        
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[-1]] = -float('inf')
        
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token])
    
    result = tokenizer.decode(tokens.tolist())
    encoder.train()
    decoder.train()
    return result


@torch.no_grad()
def generate_squad_sample(encoder, decoder, tokenizer, context, question, max_tokens, device, q_token, sep_token):
    """Generate QA answer
    
    New Architecture:
    - Encoder: Context <SEP> Question
    - Decoder: Start with <A>, generate answer with cross-attention to encoder
    """
    encoder.eval()
    decoder.eval()
    
    # Build encoder input: Context <SEP> Question
    context_tokens = tokenizer.encode(context)
    sep_marker = tokenizer.encode(sep_token)
    question_tokens = tokenizer.encode(question)
    
    encoder_input = context_tokens + sep_marker + question_tokens
    encoder_input = torch.tensor(encoder_input, dtype=torch.long, device=device)
    encoder_k, encoder_v = encoder(encoder_input, return_encoder_kv=True)
    
    # Start decoder with <A> marker
    a_marker = tokenizer.encode("<A>")
    tokens = torch.tensor(a_marker, dtype=torch.long, device=device)
    
    decoder.reset_context()
    generated = []
    for _ in range(max_tokens):
        logits = decoder(tokens, encoder_k=encoder_k, encoder_v=encoder_v)
        next_token = torch.argmax(logits[-1], dim=-1).item()
        if next_token == 0:
            break
        generated.append(next_token)
        tokens = torch.cat([tokens, torch.tensor([next_token], device=device)])
    
    result = tokenizer.decode(generated)
    encoder.train()
    decoder.train()
    return result


# ------------------------------ main ----------------------------------------
def main():
    args = get_args()
    
    # Setup
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    
    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    dtype_ctx = dtype_map[args.dtype]
    
    os.makedirs(args.out_dir, exist_ok=True)
    training_mode = TrainingMode(args.training_mode)
    
    # TensorBoard
    writer = None
    if args.use_tensorboard and TENSORBOARD_AVAILABLE:
        log_dir = os.path.join(args.log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard logging to: {log_dir}")
    
    # Tokenizer
    print("Loading tokenizer...")
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.n_vocab
    
    # Build configs
    print(f"Building {args.model_size} encoder-decoder configs...")
    encoder_config = build_config(
        args.model_size, vocab_size, is_encoder=True,
        use_lsi_compression=args.use_lsi_compression,
        num_compression_slots=args.num_compression_slots
    )
    decoder_config = build_config(args.model_size, vocab_size, is_encoder=False)
    
    # Save configs
    with open(os.path.join(args.out_dir, "encoder_config.json"), "w") as f:
        json.dump({k: v for k, v in encoder_config.__dict__.items() if not k.startswith('_')}, f, indent=2)
    with open(os.path.join(args.out_dir, "decoder_config.json"), "w") as f:
        json.dump({k: v for k, v in decoder_config.__dict__.items() if not k.startswith('_')}, f, indent=2)
    with open(os.path.join(args.out_dir, "training_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Create models
    print("Initializing encoder and decoder...")
    encoder = BidirectionalEncoder(encoder_config, device=device)
    decoder = Transformer(decoder_config, device=device)
    encoder.train()
    decoder.train()
    
    # Load pretrained weights
    pretrained_keys = []
    new_keys = []
    start_iter = 0
    
    if args.pretrained_encoder_decoder_path:
        start_iter = load_pretrained_encoder_decoder(
            encoder, decoder, args.pretrained_encoder_decoder_path, device
        )
    elif args.pretrained_decoder_path:
        pretrained_keys, new_keys = load_pretrained_decoder(
            decoder, args.pretrained_decoder_path, device
        )
    
    # Parameter count
    encoder_params = sum(p.numel() for p in encoder.parameters())
    decoder_params = sum(p.numel() for p in decoder.parameters())
    print(f"Encoder parameters: {encoder_params:,}")
    print(f"Decoder parameters: {decoder_params:,}")
    print(f"Total parameters: {encoder_params + decoder_params:,}")
    
    # Load data
    print("\n" + "="*60)
    print("Loading datasets...")
    print("="*60)
    
    # TinyStories
    print("\n--- TinyStories ---")
    stories_train_loader, stories_val_loader = create_context_dataloaders(
        batch_size=args.batch_size,
        max_length=args.block_size,
        num_workers=4,
        shuffle_train=True
    )
    stories_train_iter = iter(stories_train_loader)
    
    # SQuAD
    print("\n--- SQuAD ---")
    squad_train, squad_val = load_squad_data(
        tokenizer, args.max_context_len, args.max_answer_len,
        args.question_token, args.sep_token
    )
    
    print(f"\nTraining mode: {training_mode.value}")
    if training_mode == TrainingMode.MIXED:
        print(f"Stories weight: {args.stories_weight}, SQuAD weight: {1 - args.stories_weight}")
    elif training_mode == TrainingMode.CURRICULUM:
        print(f"Curriculum switch at iteration: {args.curriculum_switch_iter}")
    elif training_mode == TrainingMode.SEQUENTIAL:
        print(f"Sequential: {args.sequential_first} first, switch at iter {args.sequential_switch_iter}")
    
    # Setup optimizer
    print("\nSetting up optimizer...")
    decoder_lr = args.decoder_lr if args.decoder_lr is not None else args.lr
    new_layers_lr = args.new_layers_lr if args.new_layers_lr is not None else args.lr
    
    if args.pretrained_decoder_path and pretrained_keys and (decoder_lr != new_layers_lr):
        print(f"Differential learning rates:")
        print(f"  - Pretrained decoder: {decoder_lr:.6f}")
        print(f"  - Encoder + Cross-attn: {new_layers_lr:.6f}")
        
        pretrained_params = []
        new_params = []
        
        for name, param in decoder.named_parameters():
            if any(name == key for key in pretrained_keys):
                pretrained_params.append(param)
            else:
                new_params.append(param)
        
        new_params.extend(encoder.parameters())
        
        optimizer = torch.optim.AdamW([
            {'params': pretrained_params, 'lr': decoder_lr},
            {'params': new_params, 'lr': new_layers_lr},
        ], betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
        use_differential_lr = True
    else:
        all_params = list(encoder.parameters()) + list(decoder.parameters())
        optimizer = torch.optim.AdamW(
            all_params,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay
        )
        use_differential_lr = False
        print(f"Single learning rate: {args.lr:.6f}")
    
    # Resume from checkpoint
    if args.resume and args.checkpoint_path and os.path.exists(args.checkpoint_path):
        print(f"Loading checkpoint from {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_iter = checkpoint.get('iter', 0)
        print(f"Resuming from iteration {start_iter}")
    
    # Training loop
    print("\n" + "="*60)
    print("Starting multi-task training...")
    print(f"Max iterations: {args.max_iters}")
    print("="*60 + "\n")
    
    iter_num = start_iter
    running_loss = 0.0
    running_stories_loss = 0.0
    running_squad_loss = 0.0
    log_count = 0
    best_val_loss = float('inf')
    
    t0 = time.time()
    
    while iter_num < args.max_iters:
        # Determine which task to train on this iteration
        if training_mode == TrainingMode.ALTERNATING:
            train_stories = (iter_num % 2 == 0)
            train_squad = (iter_num % 2 == 1)
        elif training_mode == TrainingMode.MIXED:
            train_stories = random.random() < args.stories_weight
            train_squad = not train_stories
        elif training_mode == TrainingMode.SEQUENTIAL:
            if args.sequential_first == "stories":
                train_stories = (iter_num < args.sequential_switch_iter)
                train_squad = not train_stories
            else:
                train_squad = (iter_num < args.sequential_switch_iter)
                train_stories = not train_squad
        elif training_mode == TrainingMode.CURRICULUM:
            # Start with stories only, then mix in SQuAD
            if iter_num < args.curriculum_switch_iter:
                train_stories = True
                train_squad = False
            else:
                train_stories = random.random() < args.stories_weight
                train_squad = not train_stories
        
        # Update learning rate
        if use_differential_lr:
            pretrained_lr = get_lr(iter_num, args.warmup_iters, args.max_iters, decoder_lr, args.min_lr)
            new_lr = get_lr(iter_num, args.warmup_iters, args.max_iters, new_layers_lr, args.min_lr)
            optimizer.param_groups[0]['lr'] = pretrained_lr
            optimizer.param_groups[1]['lr'] = new_lr
            lr = new_lr
        else:
            lr = get_lr(iter_num, args.warmup_iters, args.max_iters, args.lr, args.min_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        # Training step
        optimizer.zero_grad(set_to_none=True)
        loss = None
        task_name = ""
        
        if train_stories:
            try:
                stories_batch = next(stories_train_iter)
            except StopIteration:
                stories_train_iter = iter(stories_train_loader)
                stories_batch = next(stories_train_iter)
            
            loss = train_stories_step(
                encoder, decoder, stories_batch, device, dtype_ctx, 
                args.stories_loss_weight
            )
            task_name = "stories"
            if loss is not None:
                running_stories_loss += loss.item()
        
        if train_squad:
            squad_batch = get_squad_batch(squad_train, args.batch_size, device)
            loss = train_squad_step(
                encoder, decoder, squad_batch, device, dtype_ctx,
                args.squad_loss_weight
            )
            task_name = "squad"
            if loss is not None:
                running_squad_loss += loss.item()
        
        if loss is None:
            continue
        
        # Backward
        loss.backward()
        
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()),
                args.grad_clip
            )
        
        optimizer.step()
        
        running_loss += loss.item()
        log_count += 1
        
        # Logging
        if (iter_num + 1) % args.log_interval == 0:
            avg_loss = running_loss / log_count
            t1 = time.time()
            dt = t1 - t0
            
            stories_avg = running_stories_loss / max(1, log_count // 2) if running_stories_loss > 0 else 0
            squad_avg = running_squad_loss / max(1, log_count // 2) if running_squad_loss > 0 else 0
            
            print(f"iter {iter_num + 1:5d} | loss {avg_loss:.4f} | "
                  f"stories {stories_avg:.4f} | squad {squad_avg:.4f} | "
                  f"lr {lr:.6f} | {dt*1000:.2f}ms")
            
            if writer:
                writer.add_scalar('Loss/train', avg_loss, iter_num + 1)
                writer.add_scalar('Loss/stories', stories_avg, iter_num + 1)
                writer.add_scalar('Loss/squad', squad_avg, iter_num + 1)
                writer.add_scalar('Learning_rate', lr, iter_num + 1)
            
            running_loss = 0.0
            running_stories_loss = 0.0
            running_squad_loss = 0.0
            log_count = 0
            t0 = time.time()
        
        # Evaluation
        if (iter_num + 1) % args.eval_interval == 0:
            print(f"\n--- Evaluation at iter {iter_num + 1} ---")
            
            stories_val_loss = evaluate_stories(
                encoder, decoder, stories_val_loader, 
                args.eval_iters, device, dtype_ctx
            )
            squad_val_loss = evaluate_squad(
                encoder, decoder, squad_val,
                args.eval_iters, device, dtype_ctx, args.batch_size
            )
            
            combined_val_loss = (stories_val_loss + squad_val_loss) / 2
            
            print(f"val_loss: stories={stories_val_loss:.4f}, squad={squad_val_loss:.4f}, "
                  f"combined={combined_val_loss:.4f}")
            
            if writer:
                writer.add_scalar('Loss/val_stories', stories_val_loss, iter_num + 1)
                writer.add_scalar('Loss/val_squad', squad_val_loss, iter_num + 1)
                writer.add_scalar('Loss/val_combined', combined_val_loss, iter_num + 1)
            
            if combined_val_loss < best_val_loss:
                best_val_loss = combined_val_loss
                best_path = os.path.join(args.out_dir, "best_model.pt")
                torch.save({
                    'encoder': encoder.state_dict(),
                    'decoder': decoder.state_dict(),
                    'iter': iter_num + 1,
                    'val_loss': combined_val_loss,
                    'stories_val_loss': stories_val_loss,
                    'squad_val_loss': squad_val_loss,
                }, best_path)
                print(f"Saved best model (val_loss={combined_val_loss:.4f})")
            print()
        
        # Sample generation
        if (iter_num + 1) % args.sample_every == 0:
            print(f"\n{'='*60}")
            print(f"Samples at iter {iter_num + 1}")
            print('='*60)
            
            # Story sample
            story = generate_story_sample(
                encoder, decoder, tokenizer, "Once upon a time",
                args.sample_tokens, args.temperature, args.top_k, device
            )
            print(f"\n[STORY CONTINUATION]\n{story}")
            
            # QA sample - show question and generate answer
            if squad_val:
                sample_ex = random.choice(squad_val)
                # Decode the encoder input (context + sep + question) for display
                encoder_text = tokenizer.decode(sample_ex['encoder_tokens'])
                question = sample_ex.get('question', 'N/A')
                expected_answer = sample_ex.get('answer', 'N/A')
                
                # Generate answer using the model
                generated = generate_squad_sample(
                    encoder, decoder, tokenizer,
                    tokenizer.decode(sample_ex['encoder_tokens'][:256]),  # truncated context
                    question,
                    args.sample_tokens, device,
                    args.question_token, args.sep_token
                )
                
                print(f"\n[QA SAMPLE]")
                print(f"Question: {question}")
                print(f"Expected: {expected_answer}")
                print(f"Generated: {generated}")
            
            print('='*60 + "\n")
            
            if writer:
                writer.add_text('Samples/story', story, iter_num + 1)
        
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
    print(f"\nTraining complete! Final model saved to {final_path}")
    
    if writer:
        writer.close()


if __name__ == "__main__":
    main()
