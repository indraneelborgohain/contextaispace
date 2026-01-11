#!/usr/bin/env python3
"""
train_msmarco.py - Training script for MS MARCO Reading Comprehension

Microsoft Machine Reading Comprehension (MS MARCO) dataset training.
MS MARCO is a large-scale dataset with:
- Real Bing queries as questions
- Human-generated answers (abstractive, not just extractive spans)
- Multiple passages per query

Architecture:
- Encoder: Processes context passage(s) (bidirectional attention)
- Decoder: Generates answer while attending to encoder via cross-attention
- Loss: Only computed on answer tokens (question tokens masked)
"""
import argparse
import json
import math
import os
import time
import datetime
import random

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
    ap = argparse.ArgumentParser(
        description="Train encoder-decoder on MS MARCO Reading Comprehension"
    )
    ap.add_argument("--out_dir", type=str, default="model_msmarco")
    ap.add_argument("--model_size", type=str, 
                    choices=["toy", "small", "medium", "large"], default="toy")
    
    # Training
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_context_len", type=int, default=512,
                    help="Max context/passage length")
    ap.add_argument("--max_qa_len", type=int, default=128,
                    help="Max question + answer length")
    ap.add_argument("--max_answer_len", type=int, default=64,
                    help="Max answer length for generation")
    ap.add_argument("--max_iters", type=int, default=10000)
    ap.add_argument("--log_interval", type=int, default=10)
    ap.add_argument("--eval_interval", type=int, default=200)
    ap.add_argument("--eval_iters", type=int, default=20)
    
    # Save + sample
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--sample_every", type=int, default=500)
    
    # Optimizer
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--warmup_iters", type=int, default=200)
    ap.add_argument("--min_lr", type=float, default=3e-5)
    
    # System
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--dtype", type=str, 
                    choices=["float32", "bfloat16", "float16"], default="bfloat16")
    
    # Checkpoint
    ap.add_argument("--resume", action="store_true", default=False)
    ap.add_argument("--checkpoint_path", type=str, default=None)
    
    # Pretrained decoder
    ap.add_argument("--pretrained_decoder_path", type=str, default=None,
                    help="Path to pretrained decoder weights")
    ap.add_argument("--pretrained_encoder_decoder_path", type=str, default=None,
                    help="Path to pretrained encoder-decoder checkpoint")
    ap.add_argument("--decoder_lr", type=float, default=None,
                    help="Learning rate for pretrained decoder (if None, uses --lr)")
    ap.add_argument("--new_layers_lr", type=float, default=None,
                    help="Learning rate for encoder and cross-attention (if None, uses --lr)")
    
    # TensorBoard
    ap.add_argument("--use_tensorboard", action="store_true", default=False)
    ap.add_argument("--log_dir", type=str, default="runs_msmarco")
    
    # Special tokens
    ap.add_argument("--question_token", type=str, default="<Q>")
    ap.add_argument("--answer_token", type=str, default="<A>")
    ap.add_argument("--sep_token", type=str, default="<SEP>")
    
    # Encoder compression
    ap.add_argument("--use_lsi_compression", action="store_true", default=False,
                    help="Use LSI cross-attention for encoder compression")
    ap.add_argument("--num_compression_slots", type=int, default=64,
                    help="Number of latent slots for LSI compression")
    
    # MS MARCO specific
    ap.add_argument("--dataset_version", type=str, default="v2.1",
                    choices=["v1.1", "v2.1"],
                    help="MS MARCO dataset version")
    ap.add_argument("--use_all_passages", action="store_true", default=False,
                    help="Concatenate all passages (vs just selected passage)")
    ap.add_argument("--max_passages", type=int, default=3,
                    help="Max number of passages to use if use_all_passages")
    ap.add_argument("--skip_no_answer", action="store_true", default=True,
                    help="Skip examples with 'No Answer Present' label")
    
    return ap.parse_args()


# ------------------------------ pretrained loading --------------------------
def load_pretrained_decoder(decoder, pretrained_path, device):
    """Load pretrained decoder weights with partial loading support"""
    print(f"Loading pretrained decoder from {pretrained_path}...")
    checkpoint = torch.load(pretrained_path, map_location=device, weights_only=False)
    
    if 'model' in checkpoint:
        pretrained_state = checkpoint['model']
    elif 'decoder' in checkpoint:
        pretrained_state = checkpoint['decoder']
    else:
        pretrained_state = checkpoint
    
    model_state = decoder.state_dict()
    loaded_keys = []
    skipped_keys = []
    
    for name, param in pretrained_state.items():
        if 'cross_attn' in name:
            skipped_keys.append(f"{name} (cross-attention)")
            continue
        if name in model_state:
            if param.shape == model_state[name].shape:
                model_state[name] = param
                loaded_keys.append(name)
            else:
                skipped_keys.append(f"{name} (shape mismatch)")
        else:
            skipped_keys.append(f"{name} (not in model)")
    
    new_keys = [name for name in model_state.keys() if name not in pretrained_state]
    decoder.load_state_dict(model_state)
    
    print(f"✓ Loaded {len(loaded_keys)} parameters")
    print(f"✗ Skipped {len(skipped_keys)} parameters")
    print(f"✓ Initialized {len(new_keys)} new parameters")
    
    return loaded_keys, new_keys


def load_pretrained_encoder_decoder(encoder, decoder, checkpoint_path, device):
    """Load both encoder and decoder from a checkpoint"""
    print(f"Loading pretrained encoder-decoder from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'encoder' in checkpoint:
        encoder.load_state_dict(checkpoint['encoder'])
        print(f"✓ Loaded encoder")
    
    if 'decoder' in checkpoint:
        decoder.load_state_dict(checkpoint['decoder'])
        print(f"✓ Loaded decoder")
    
    return checkpoint.get('iter', 0)


# ------------------------------ config --------------------------------------
def build_config(name: str, vocab_size: int, use_lsi_compression: bool = False, 
                 num_compression_slots: int = 64):
    """Build encoder and decoder configs based on size"""
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
    
    return configs[name]["encoder"], configs[name]["decoder"]


# ------------------------------ data ----------------------------------------
def format_msmarco_example(example, tokenizer, max_context_len, max_qa_len, 
                           q_token, a_token, sep_token, use_all_passages, 
                           max_passages, skip_no_answer):
    """
    Format a single MS MARCO example for encoder-decoder training.
    
    MS MARCO format:
    - query: The question
    - passages: Dict with 'passage_text' list and 'is_selected' list
    - answers: List of answers (can have multiple or be empty)
    - wellFormedAnswers: List of well-formed answers (optional)
    
    Returns:
        Dict with context_tokens, qa_tokens, target_tokens, or None if invalid
    """
    query = example.get('query', '')
    
    # Get passages
    passages_data = example.get('passages', {})
    passage_texts = passages_data.get('passage_text', [])
    is_selected = passages_data.get('is_selected', [])
    
    # Get answers
    answers = example.get('answers', [])
    well_formed = example.get('wellFormedAnswers', [])
    
    # Skip if no valid answer
    if not answers or (len(answers) == 1 and answers[0].lower() == 'no answer present.'):
        if skip_no_answer:
            return None
        # Use a placeholder for no-answer cases
        answer = "No answer available."
    else:
        # Prefer well-formed answers if available
        if well_formed and len(well_formed) > 0 and well_formed[0]:
            answer = well_formed[0]
        else:
            answer = answers[0]
    
    # Build context from passages
    if use_all_passages and len(passage_texts) > 1:
        # Concatenate multiple passages
        selected_passages = []
        
        # First add selected passages
        for i, (text, selected) in enumerate(zip(passage_texts, is_selected)):
            if selected == 1 and len(selected_passages) < max_passages:
                selected_passages.append(text)
        
        # If not enough, add others
        for i, (text, selected) in enumerate(zip(passage_texts, is_selected)):
            if selected != 1 and len(selected_passages) < max_passages:
                selected_passages.append(text)
        
        context = " [SEP] ".join(selected_passages)
    else:
        # Use only selected passage(s)
        selected_passages = [
            text for text, selected in zip(passage_texts, is_selected) 
            if selected == 1
        ]
        if selected_passages:
            context = selected_passages[0]
        elif passage_texts:
            context = passage_texts[0]  # Fallback to first passage
        else:
            return None  # No passages available
    
    # Tokenize
    context_tokens = tokenizer.encode(context)
    query_tokens = tokenizer.encode(query)
    answer_tokens = tokenizer.encode(answer)
    
    c_marker = tokenizer.encode("<C>")
    a_marker = tokenizer.encode(a_token)
    sep_marker = tokenizer.encode(sep_token)
    
    # Build encoder input: <C> Context <SEP> Query
    # Reserve space for query, separator, and context marker
    max_context_space = max_context_len - len(query_tokens) - len(sep_marker) - len(c_marker)
    if max_context_space < 50:  # Need at least some context
        return None
    
    if len(context_tokens) > max_context_space:
        context_tokens = context_tokens[:max_context_space]
    
    encoder_tokens = c_marker + context_tokens + sep_marker + query_tokens
    
    # Build decoder input: <A> Answer (teacher forcing)
    if len(answer_tokens) > max_qa_len - len(a_marker):
        answer_tokens = answer_tokens[:max_qa_len - len(a_marker)]
    
    decoder_tokens = a_marker + answer_tokens
    
    # Target: Answer tokens (for next-token prediction)
    target_tokens = answer_tokens + [0]  # Add EOS token
    
    return {
        'encoder_tokens': encoder_tokens,
        'decoder_tokens': decoder_tokens,
        'target_tokens': target_tokens,
        'query': query,
        'answer': answer,
    }


def load_msmarco_data(tokenizer, args):
    """Load and format MS MARCO dataset"""
    print("Loading MS MARCO dataset...")
    print("(This may take a while on first download)")
    
    # Load MS MARCO from Hugging Face
    # The dataset is large, so we use streaming for efficiency
    try:
        dataset = load_dataset("ms_marco", args.dataset_version)
    except Exception as e:
        print(f"Error loading ms_marco: {e}")
        print("Trying alternative dataset name...")
        try:
            dataset = load_dataset("microsoft/ms_marco", args.dataset_version)
        except Exception as e2:
            print(f"Error: {e2}")
            print("Please ensure you have access to MS MARCO dataset.")
            print("You may need to accept terms at: https://huggingface.co/datasets/ms_marco")
            raise
    
    print("Formatting examples...")
    train_examples = []
    val_examples = []
    
    # Process training set
    train_data = dataset.get('train', [])
    for example in train_data:
        formatted = format_msmarco_example(
            example, tokenizer, args.max_context_len, args.max_qa_len,
            args.question_token, args.answer_token, args.sep_token,
            args.use_all_passages, args.max_passages, args.skip_no_answer
        )
        if formatted is not None:
            train_examples.append(formatted)
        
        # Limit for memory (MS MARCO is very large)
        if len(train_examples) >= 100000:
            print("Capped training examples at 100,000")
            break
    
    # Process validation set
    val_data = dataset.get('validation', dataset.get('dev', []))
    for example in val_data:
        formatted = format_msmarco_example(
            example, tokenizer, args.max_context_len, args.max_qa_len,
            args.question_token, args.answer_token, args.sep_token,
            args.use_all_passages, args.max_passages, args.skip_no_answer
        )
        if formatted is not None:
            val_examples.append(formatted)
        
        if len(val_examples) >= 10000:
            print("Capped validation examples at 10,000")
            break
    
    print(f"Training examples: {len(train_examples)}")
    print(f"Validation examples: {len(val_examples)}")
    
    return train_examples, val_examples


def get_batch(examples, batch_size, device):
    """Get a random batch of examples"""
    batch = random.sample(examples, min(batch_size, len(examples)))
    
    max_encoder = max(len(ex['encoder_tokens']) for ex in batch)
    max_decoder = max(len(ex['decoder_tokens']) for ex in batch)
    max_target = max(len(ex['target_tokens']) for ex in batch)
    
    encoder_batch = []
    decoder_batch = []
    target_batch = []
    
    for ex in batch:
        # Pad encoder (context + question)
        encoder = ex['encoder_tokens'] + [0] * (max_encoder - len(ex['encoder_tokens']))
        encoder_batch.append(encoder)
        
        # Pad decoder (<A> + answer)
        decoder = ex['decoder_tokens'] + [0] * (max_decoder - len(ex['decoder_tokens']))
        decoder_batch.append(decoder)
        
        # Pad targets
        target = ex['target_tokens'] + [-100] * (max_target - len(ex['target_tokens']))
        target_batch.append(target)
    
    return (
        torch.tensor(encoder_batch, dtype=torch.long, device=device),
        torch.tensor(decoder_batch, dtype=torch.long, device=device),
        torch.tensor(target_batch, dtype=torch.long, device=device),
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
        encoder_batch, decoder_batch, target_batch = get_batch(val_examples, batch_size, device)
        
        batch_loss = 0.0
        for i in range(encoder_batch.shape[0]):
            encoder_tokens = encoder_batch[i]
            decoder_tokens = decoder_batch[i]
            targets = target_batch[i]
            
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
                
                # Compute loss (predict next token)
                loss = F.cross_entropy(
                    logits[:-1].view(-1, logits.size(-1)),
                    targets[:len(logits)-1].view(-1),
                    ignore_index=-100
                )
            
            batch_loss += loss.item()
        
        losses.append(batch_loss / encoder_batch.shape[0])
    
    encoder.train()
    decoder.train()
    
    return sum(losses) / len(losses) if losses else float('inf')


@torch.no_grad()
def generate_answer(encoder, decoder, tokenizer, context, question, max_tokens, 
                    device, q_token, a_token, sep_token, temperature=1.0, top_k=50):
    """Generate answer for a given context and question"""
    encoder.eval()
    decoder.eval()
    
    # Build encoder input: <C> context <SEP> question
    c_marker = tokenizer.encode("<C>")
    context_tokens = tokenizer.encode(context)
    question_tokens = tokenizer.encode(question)
    sep_marker = tokenizer.encode(sep_token)
    
    encoder_tokens = c_marker + context_tokens + sep_marker + question_tokens
    encoder_tokens = torch.tensor(encoder_tokens, dtype=torch.long, device=device)
    
    # Encode context + question
    encoder_k, encoder_v = encoder(encoder_tokens, return_encoder_kv=True)
    
    # Start with <A> token
    a_marker = tokenizer.encode(a_token)
    tokens = torch.tensor(a_marker, dtype=torch.long, device=device)
    
    # Generate answer
    decoder.reset_context()
    generated = []
    
    for _ in range(max_tokens):
        logits = decoder(tokens, encoder_k=encoder_k, encoder_v=encoder_v)
        logits = logits[-1] / temperature
        
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[-1]] = -float('inf')
        
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        generated.append(next_token.item())
        tokens = torch.cat([tokens, next_token])
        
        # Stop at end token or newline
        if next_token.item() == 0:
            break
    
    answer = tokenizer.decode(generated)
    
    encoder.train()
    decoder.train()
    
    return answer


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
    print(f"Building {args.model_size} model configs...")
    encoder_config, decoder_config = build_config(
        args.model_size, vocab_size,
        use_lsi_compression=args.use_lsi_compression,
        num_compression_slots=args.num_compression_slots
    )
    
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
    
    # Count parameters
    encoder_params = sum(p.numel() for p in encoder.parameters())
    decoder_params = sum(p.numel() for p in decoder.parameters())
    print(f"Encoder parameters: {encoder_params:,}")
    print(f"Decoder parameters: {decoder_params:,}")
    print(f"Total parameters: {encoder_params + decoder_params:,}")
    
    # Load MS MARCO data
    train_examples, val_examples = load_msmarco_data(tokenizer, args)
    
    if len(train_examples) == 0:
        print("ERROR: No training examples loaded!")
        return
    
    # Setup optimizer
    print("Setting up optimizer...")
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
    print("Starting MS MARCO training...")
    print(f"Max iterations: {args.max_iters}")
    print("="*60 + "\n")
    
    iter_num = start_iter
    running_loss = 0.0
    log_count = 0
    best_val_loss = float('inf')
    
    t0 = time.time()
    
    while iter_num < args.max_iters:
        # Get batch
        encoder_batch, decoder_batch, target_batch = get_batch(
            train_examples, args.batch_size, device
        )
        
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
        
        # Forward pass
        total_loss = 0.0
        for i in range(encoder_batch.shape[0]):
            encoder_tokens = encoder_batch[i]
            decoder_tokens = decoder_batch[i]
            targets = target_batch[i]
            
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
                
                # Compute loss (predict next token)
                loss = F.cross_entropy(
                    logits[:-1].view(-1, logits.size(-1)),
                    targets[:len(logits)-1].view(-1),
                    ignore_index=-100
                )
            
            total_loss += loss
        
        # Average loss over batch
        loss = total_loss / encoder_batch.shape[0]
        
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
        
        # Logging
        running_loss += loss.item()
        log_count += 1
        
        if (iter_num + 1) % args.log_interval == 0:
            avg_loss = running_loss / log_count
            t1 = time.time()
            dt = t1 - t0
            print(f"iter {iter_num + 1:5d} | loss {avg_loss:.4f} | lr {lr:.6f} | {dt*1000:.2f}ms")
            
            if writer:
                writer.add_scalar('Loss/train', avg_loss, iter_num + 1)
                writer.add_scalar('Learning_rate', lr, iter_num + 1)
            
            running_loss = 0.0
            log_count = 0
            t0 = time.time()
        
        # Evaluation
        if (iter_num + 1) % args.eval_interval == 0:
            val_loss = evaluate(
                encoder, decoder, val_examples, 
                args.eval_iters, device, dtype_ctx, args.batch_size
            )
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
            sample_ex = random.choice(val_examples)
            
            # Get query and answer (with fallback for missing keys)
            query = sample_ex.get('query', sample_ex.get('question', 'N/A'))
            answer = sample_ex.get('answer', 'N/A')
            
            # Decode encoder tokens to get context for display
            # The encoder_tokens contain: context <SEP> question
            encoder_text = tokenizer.decode(sample_ex['encoder_tokens'][:256])
            
            generated = generate_answer(
                encoder, decoder, tokenizer,
                encoder_text, query, args.max_answer_len,
                device, args.question_token, args.answer_token, args.sep_token
            )
            
            print(f"\n{'='*60}")
            print(f"Sample at iter {iter_num + 1}:")
            print(f"Query: {query}")
            print(f"Gold Answer: {answer}")
            print(f"Generated: {generated}")
            print(f"{'='*60}\n")
            
            if writer:
                writer.add_text('Samples/query', query, iter_num + 1)
                writer.add_text('Samples/generated', generated, iter_num + 1)
                writer.add_text('Samples/gold', answer, iter_num + 1)
        
        # Save checkpoint
        if (iter_num + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.out_dir, f"checkpoint_{iter_num + 1}.pt")
            torch.save({
                'encoder': encoder.state_dict(),
                'decoder': encoder.state_dict(),
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
