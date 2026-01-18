#!/usr/bin/env python3
"""
Clean training script for encoder-decoder model.

Clear 3-step process:
1. Create encoder config from BERT
2. Load BERT weights into encoder  
3. Create decoder from GPT-OSS weights
"""
import argparse
import os
import time
import torch
import torch.nn.functional as F
from datasets import load_dataset

from architecture.config import ModelConfig
from architecture.encoder import BidirectionalEncoder, create_encoder_config_from_bert, load_bert_encoder
from architecture.transformer import Transformer
from architecture.tokenizer import get_tokenizer


def load_gptoss_decoder(decoder_config, gptoss_weights_dir, device):
    """
    STEP 3: Load GPT-OSS weights into decoder.
    """
    import glob
    
    print(f"\n{'='*60}")
    print("STEP 3: Loading GPT-OSS weights into decoder")
    print(f"{'='*60}")
    print(f"Weights directory: {gptoss_weights_dir}")
    
    # Find safetensors files
    safetensor_files = glob.glob(os.path.join(gptoss_weights_dir, "*.safetensors"))
    
    if not safetensor_files:
        print("⚠️  No .safetensors files found")
        print("Creating decoder with random weights...")
        return Transformer(decoder_config, device=device)
    
    # Load GPT-OSS state
    try:
        from safetensors.torch import load_file
        gptoss_state = {}
        for f in safetensor_files:
            gptoss_state.update(load_file(f))
        print(f"✓ Loaded {len(safetensor_files)} safetensors files")
    except ImportError:
        print("⚠️  safetensors not installed")
        print("Install with: pip install safetensors")
        return Transformer(decoder_config, device=device)
    
    # Create decoder
    decoder = Transformer(decoder_config, device=device)
    
    # Load weights manually
    loaded_count = 0
    
    # 1. Load embedding
    for emb_key in ['wte.weight', 'transformer.wte.weight', 'embedding.weight']:
        if emb_key in gptoss_state:
            gpt_emb = gptoss_state[emb_key]
            dec_emb = decoder.embedding.weight
            min_vocab = min(gpt_emb.size(0), dec_emb.size(0))
            if gpt_emb.size(1) == dec_emb.size(1):
                decoder.embedding.weight.data[:min_vocab] = gpt_emb[:min_vocab].to(device)
                loaded_count += 1
                print(f"✓ Loaded embeddings ({min_vocab} tokens)")
                break
    
    # 2. Load transformer layers
    num_layers = decoder_config.num_hidden_layers
    print(f"Loading {num_layers} transformer layers...")
    
    for layer_idx in range(num_layers):
        # Try different naming conventions
        for gpt_prefix in [f'layers.{layer_idx}', f'transformer.h.{layer_idx}', f'h.{layer_idx}']:
            if f'{gpt_prefix}.attn.c_attn.weight' not in gptoss_state:
                continue
            
            dec_prefix = f'block.{layer_idx}'
            
            # Load layer norm 1
            if f'{gpt_prefix}.ln_1.weight' in gptoss_state:
                decoder.state_dict()[f'{dec_prefix}.ln1.weight'].copy_(gptoss_state[f'{gpt_prefix}.ln_1.weight'])
                decoder.state_dict()[f'{dec_prefix}.ln1.bias'].copy_(gptoss_state[f'{gpt_prefix}.ln_1.bias'])
                loaded_count += 2
            
            # Load attention QKV (fused)
            if f'{gpt_prefix}.attn.c_attn.weight' in gptoss_state:
                decoder.state_dict()[f'{dec_prefix}.attn.qkv.weight'].copy_(gptoss_state[f'{gpt_prefix}.attn.c_attn.weight'])
                decoder.state_dict()[f'{dec_prefix}.attn.qkv.bias'].copy_(gptoss_state[f'{gpt_prefix}.attn.c_attn.bias'])
                loaded_count += 2
            
            # Load attention output
            if f'{gpt_prefix}.attn.c_proj.weight' in gptoss_state:
                decoder.state_dict()[f'{dec_prefix}.attn.out.weight'].copy_(gptoss_state[f'{gpt_prefix}.attn.c_proj.weight'])
                decoder.state_dict()[f'{dec_prefix}.attn.out.bias'].copy_(gptoss_state[f'{gpt_prefix}.attn.c_proj.bias'])
                loaded_count += 2
            
            print(f"  ✓ Layer {layer_idx}")
            break
    
    print(f"\n✓ Successfully loaded {loaded_count} parameter groups from GPT-OSS")
    print(f"Decoder parameters: {sum(p.numel() for p in decoder.parameters())/1e6:.1f}M\n")
    
    return decoder


def get_lr(it, warmup_iters, max_iters, learning_rate, min_lr):
    """Learning rate schedule with warmup and cosine decay."""
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + torch.cos(torch.tensor(decay_ratio * 3.14159)))
    return min_lr + coeff * (learning_rate - min_lr)


def load_saved_model(model_dir, device):
    """Load encoder and decoder from saved checkpoint."""
    print(f"\n{'='*60}")
    print("LOADING SAVED MODEL")
    print(f"{'='*60}")
    print(f"Model directory: {model_dir}")
    
    encoder_path = os.path.join(model_dir, "encoder.pt")
    decoder_path = os.path.join(model_dir, "decoder.pt")
    
    if not os.path.exists(encoder_path) or not os.path.exists(decoder_path):
        print(f"⚠️  Model files not found in {model_dir}")
        return None, None
    
    try:
        # Load encoder
        print(f"Loading encoder from {encoder_path}...")
        encoder_checkpoint = torch.load(encoder_path, map_location=device)
        encoder_config = encoder_checkpoint['config']
        encoder = BidirectionalEncoder(encoder_config, device=device)
        encoder.load_state_dict(encoder_checkpoint['model_state_dict'])
        print(f"✓ Encoder loaded ({sum(p.numel() for p in encoder.parameters())/1e6:.1f}M params)")
        
        # Load decoder
        print(f"Loading decoder from {decoder_path}...")
        decoder_checkpoint = torch.load(decoder_path, map_location=device)
        decoder_config = decoder_checkpoint['config']
        decoder = Transformer(decoder_config, device=device)
        decoder.load_state_dict(decoder_checkpoint['model_state_dict'])
        print(f"✓ Decoder loaded ({sum(p.numel() for p in decoder.parameters())/1e6:.1f}M params)")
        
        print(f"{'='*60}\n")
        return encoder, decoder
        
    except Exception as e:
        print(f"⚠️  Error loading model: {e}")
        return None, None


def main():
    parser = argparse.ArgumentParser(description="Clean training script")
    parser.add_argument("--out_dir", type=str, default="model_clean", help="Output directory")
    parser.add_argument("--model_dir", type=str, default=None, help="Directory with saved model to resume from")
    args = parser.parse_args()
    
    # ========================================
    # Configuration
    # ========================================
    bert_model_name = "bert-large-uncased"
    gptoss_weights_dir = "architecture/open-gpt-oss/weights"
    
    batch_size = 1
    gradient_accumulation_steps = 4
    max_context_len = 256
    max_qa_len = 128
    max_iters = 10000
    eval_interval = 100
    save_every = 500
    log_interval = 10
    
    encoder_lr = 1e-5
    decoder_lr = 1e-6
    cross_attn_lr = 3e-4
    warmup_iters = 200
    min_lr_ratio = 0.1
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    seed = 42
    
    # Setup
    torch.manual_seed(seed)
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("ENCODER-DECODER TRAINING")
    print("="*60)
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print(f"Batch size: {batch_size}")
    print(f"Gradient accumulation: {gradient_accumulation_steps}")
    print(f"Effective batch: {batch_size * gradient_accumulation_steps}")
    print("="*60 + "\n")
    
    # Get tokenizer
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.n_vocab
    
    # Check if loading from saved model
    if args.model_dir and os.path.exists(args.model_dir):
        encoder, decoder = load_saved_model(args.model_dir, device)
        if encoder is not None and decoder is not None:
            # Successfully loaded - skip BERT and GPT-OSS loading
            encoder_config = encoder.config
            decoder_config = decoder.config
            print("✓ Skipping BERT and GPT-OSS loading (using saved model)\n")
        else:
            # Failed to load - proceed with normal initialization
            encoder = None
            decoder = None
    else:
        encoder = None
        decoder = None
    
    # Only create new models if we didn't load from checkpoint
    if encoder is None or decoder is None:
        # ========================================
        # STEP 1: Create encoder config from BERT
        # ========================================
        print("="*60)
        print("STEP 1: Creating encoder config from BERT")
        print("="*60)
        print(f"BERT model: {bert_model_name}")
        
        encoder_config = create_encoder_config_from_bert(bert_model_name)
    
        # Override vocab size to match our tokenizer
        encoder_config.vocab_size = vocab_size
        
        print(f"\n✓ Encoder config created:")
        print(f"  Vocab size: {encoder_config.vocab_size}")
        print(f"  Hidden size: {encoder_config.hidden_size}")
        print(f"  Layers: {encoder_config.num_hidden_layers}")
        print(f"  Attention heads: {encoder_config.num_attention_heads}")
        print(f"  Intermediate size: {encoder_config.intermediate_size}")
        print(f"  Max position: {encoder_config.max_position_embeddings}\n")
        
        # ========================================
        # STEP 2: Load BERT weights into encoder
        # ========================================
        print("="*60)
        print("STEP 2: Loading BERT weights into encoder")
        print("="*60)
        
        encoder = load_bert_encoder(encoder_config, bert_model_name, device)
        
        if encoder is None:
            print("⚠️  Failed to load BERT weights")
            print("Creating encoder with random weights...")
            encoder = BidirectionalEncoder(encoder_config, device=device)
        
        print(f"\n✓ Encoder ready")
        print(f"Encoder parameters: {sum(p.numel() for p in encoder.parameters())/1e6:.1f}M\n")
        
        # ========================================
        # STEP 3: Create decoder from GPT-OSS
        # ========================================
        # Using SMALLER decoder config for GPU memory constraints
        # Original GPT-OSS: 2880 hidden, 12 layers, 64 heads, 32 experts
        # Smaller config: 768 hidden, 6 layers, 12 heads, 8 experts
        decoder_config = ModelConfig(
            vocab_size=vocab_size,
            hidden_size=768,          # Reduced from 2880
            num_hidden_layers=6,      # Reduced from 12
            num_experts=8,            # Reduced from 32
            num_attention_heads=12,   # Reduced from 64
            num_key_value_heads=12,   # Reduced from 64
            use_encoder_decoder_cross_attention=True,
            encoder_hidden_size=encoder_config.hidden_size,
            use_context_embedding=True,  # Enable context state tracking
            sliding_window=512           # Set sliding window size
        )
        
        print(f"Decoder config:")
        print(f"  Vocab size: {decoder_config.vocab_size}")
        print(f"  Hidden size: {decoder_config.hidden_size}")
        print(f"  Layers: {decoder_config.num_hidden_layers}")
        print(f"  Cross-attention: {decoder_config.use_encoder_decoder_cross_attention}")
        print(f"  Encoder hidden size: {decoder_config.encoder_hidden_size}")
        
        decoder = load_gptoss_decoder(decoder_config, gptoss_weights_dir, device)
    
    # ========================================
    # Training setup
    # ========================================
    print("="*60)
    print("TRAINING SETUP")
    print("="*60)
    
    # Setup optimizer with 3-tier learning rates
    encoder_params = list(encoder.parameters())
    
    # Separate decoder params
    decoder_base_params = []
    decoder_custom_params = []
    for name, param in decoder.named_parameters():
        if 'context_proj' in name or 'cross_attn' in name:
            decoder_custom_params.append(param)
        else:
            decoder_base_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': encoder_lr, 'name': 'encoder'},
        {'params': decoder_base_params, 'lr': decoder_lr, 'name': 'decoder_base'},
        {'params': decoder_custom_params, 'lr': cross_attn_lr, 'name': 'cross_attn'}
    ], betas=(0.9, 0.95), weight_decay=0.1)
    
    print("✓ Optimizer created with 3-tier learning rates:")
    print(f"  Encoder: {encoder_lr}")
    print(f"  Decoder base: {decoder_lr}")
    print(f"  Cross-attention: {cross_attn_lr}\n")
    
    # Load dataset
    print("Loading MS MARCO dataset...")
    dataset = load_dataset("ms_marco", "v2.1")
    
    # Prepare training examples
    train_examples = []
    for example in dataset['train']:
        if not example.get('passages') or not example['passages'].get('passage_text'):
            continue
        if not example.get('query'):
            continue
        if not example.get('answers') or len(example['answers']) == 0:
            continue
        
        context = example['passages']['passage_text'][0]
        question = example['query']
        answer = example['answers'][0]
        
        train_examples.append({
            'context': context,
            'question': question,
            'answer': answer
        })
        
        if len(train_examples) >= 50000:
            break
    
    print(f"✓ Loaded {len(train_examples)} training examples\n")
    
    # Prepare validation examples
    val_examples = []
    for example in dataset['validation']:
        if not example.get('passages') or not example['passages'].get('passage_text'):
            continue
        if not example.get('query'):
            continue
        if not example.get('answers') or len(example['answers']) == 0:
            continue
        
        context = example['passages']['passage_text'][0]
        question = example['query']
        answer = example['answers'][0]
        
        val_examples.append({
            'context': context,
            'question': question,
            'answer': answer
        })
        
        if len(val_examples) >= 1000:
            break
    
    print(f"✓ Loaded {len(val_examples)} validation examples\n")
    
    # Training loop
    print("="*60)
    print("STARTING TRAINING")
    print("="*60 + "\n")
    
    encoder.train()
    decoder.train()
    
    best_val_loss = float('inf')
    train_idx = 0
    sep_token_id = tokenizer.encode("<SEP>", allowed_special={'<SEP>'})[0]
    
    for it in range(max_iters):
        # Get learning rates
        enc_lr = get_lr(it, warmup_iters, max_iters, encoder_lr, encoder_lr * min_lr_ratio)
        dec_lr = get_lr(it, warmup_iters, max_iters, decoder_lr, decoder_lr * min_lr_ratio)
        cross_lr = get_lr(it, warmup_iters, max_iters, cross_attn_lr, cross_attn_lr * min_lr_ratio)
        
        optimizer.param_groups[0]['lr'] = enc_lr
        optimizer.param_groups[1]['lr'] = dec_lr
        optimizer.param_groups[2]['lr'] = cross_lr
        
        # Gradient accumulation
        total_loss = 0.0
        for micro_step in range(gradient_accumulation_steps):
            # CRITICAL: Reset context before each example to prevent bleeding
            decoder.reset_context()
            
            # Get batch
            example = train_examples[train_idx % len(train_examples)]
            train_idx += 1
            
            context = example['context'][:max_context_len*4]
            question = example['question'][:max_qa_len]
            answer = example['answer'][:max_qa_len]
            
            # Tokenize
            context_tokens = tokenizer.encode(context)[:max_context_len]
            question_tokens = tokenizer.encode(question)[:max_qa_len]
            answer_tokens = tokenizer.encode(answer)[:max_qa_len]
            # Add EOS token to answer
            end_token_id = tokenizer.encode("<|endoftext|>", allowed_special={'<|endoftext|>'})[0]
            answer_tokens = answer_tokens + [end_token_id]
            
            # Create encoder input: context <SEP> question
            encoder_input = torch.tensor(
                context_tokens + [sep_token_id] + question_tokens,
                dtype=torch.long,
                device=device
            )
            
            # Create decoder input/target
            decoder_input = torch.tensor(answer_tokens[:-1], dtype=torch.long, device=device)
            decoder_target = torch.tensor(answer_tokens[1:], dtype=torch.long, device=device)
            
            # Forward pass
            with torch.amp.autocast(device_type='cuda', dtype=dtype):
                # Encode
                encoder_k, encoder_v = encoder(
                    encoder_input,
                    return_encoder_kv=True,
                    sep_token_id=sep_token_id
                )
                
                # Decode
                logits = decoder(
                    decoder_input,
                    encoder_k=encoder_k,
                    encoder_v=encoder_v,
                    update_context=True  # Allow context updates during training
                )
                
                # Loss
                loss = F.cross_entropy(
                    logits.view(-1, vocab_size),
                    decoder_target.view(-1)
                )
                loss = loss / gradient_accumulation_steps
            
            # Backward
            loss.backward()
            total_loss += loss.item()
        
        # Optimizer step
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        # Logging
        if it % log_interval == 0:
            # Add context state diagnostic
            context_norm = decoder.context_state.norm().item() if hasattr(decoder, 'context_state') else 0.0
            print(f"Iter {it:5d} | Loss: {total_loss:.4f} | LR: {enc_lr:.2e}/{dec_lr:.2e}/{cross_lr:.2e} | Ctx: {context_norm:.4f}")
        
        # Validation
        if it > 0 and it % eval_interval == 0:
            encoder.eval()
            decoder.eval()
            
            val_loss = 0.0
            num_val = min(10, len(val_examples))
            
            with torch.no_grad():
                for val_idx in range(num_val):
                    # CRITICAL: Reset context for each validation example
                    decoder.reset_context()
                    
                    example = val_examples[val_idx]
                    
                    context = example['context'][:max_context_len*4]
                    question = example['question'][:max_qa_len]
                    answer = example['answer'][:max_qa_len]
                    
                    # Tokenize
                    context_tokens = tokenizer.encode(context)[:max_context_len]
                    question_tokens = tokenizer.encode(question)[:max_qa_len]
                    answer_tokens = tokenizer.encode(answer)[:max_qa_len]
                    # Add EOS token to answer
                    end_token_id = tokenizer.encode("<|endoftext|>", allowed_special={'<|endoftext|>'})[0]
                    answer_tokens = answer_tokens + [end_token_id]
                    
                    # Encode
                    encoder_input = torch.tensor(
                        context_tokens + [sep_token_id] + question_tokens,
                        dtype=torch.long,
                        device=device
                    )
                    decoder_input = torch.tensor(answer_tokens[:-1], dtype=torch.long, device=device)
                    decoder_target = torch.tensor(answer_tokens[1:], dtype=torch.long, device=device)
                    
                    with torch.amp.autocast(device_type='cuda', dtype=dtype):
                        encoder_k, encoder_v = encoder(encoder_input, return_encoder_kv=True, sep_token_id=sep_token_id)
                        logits = decoder(decoder_input, encoder_k=encoder_k, encoder_v=encoder_v)
                        loss = F.cross_entropy(logits.view(-1, vocab_size), decoder_target.view(-1))
                    
                    val_loss += loss.item()
            
            val_loss /= num_val
            
            print(f"\n{'='*60}")
            print(f"VALIDATION @ Iter {it}")
            print(f"{'='*60}")
            print(f"Val Loss: {val_loss:.4f}")
            
            # Generate sample
            example = val_examples[0]
            print(f"\nSample Generation:")
            print(f"Context: {example['context'][:200]}...")
            print(f"Question: {example['question']}")
            print(f"Ground Truth: {example['answer']}")
            
            # Generate
            context_tokens = tokenizer.encode(example['context'][:max_context_len*4])[:max_context_len]
            question_tokens = tokenizer.encode(example['question'][:max_qa_len])[:max_qa_len]
            encoder_input = torch.tensor(context_tokens + [sep_token_id] + question_tokens, dtype=torch.long, device=device)
            
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda', dtype=dtype):
                    # Encode
                    encoder_k, encoder_v = encoder(encoder_input, return_encoder_kv=True, sep_token_id=sep_token_id)
                    
                    # CRITICAL: Reset decoder context before generation
                    decoder.reset_context()
                    
                    # Generate token by token
                    generated = []
                    a_token_id = tokenizer.encode("<A>", allowed_special={'<A>'})[0]
                    end_token_id = tokenizer.encode("<|endoftext|>", allowed_special={'<|endoftext|>'})[0]
                    
                    current_token = a_token_id
                    max_gen_len = 64
                    temperature = 0.8
                    
                    for step in range(max_gen_len):
                        # CRITICAL: Pass single token, let decoder handle context internally
                        token_input = torch.tensor([current_token], dtype=torch.long, device=device)
                        
                        logits = decoder(
                            token_input,
                            encoder_k=encoder_k,
                            encoder_v=encoder_v,
                            update_context=True  # Update context after each token
                        )
                        
                        # Get next token with temperature sampling
                        next_logits = logits[-1] / temperature
                        probs = F.softmax(next_logits, dim=-1)
                        
                        # Sample instead of argmax for better diversity
                        next_token = torch.multinomial(probs, num_samples=1).item()
                        
                        # Debug first few tokens
                        if step < 5:
                            print(f"  Gen step {step}: token={next_token}, prob={probs[next_token]:.4f}")
                        
                        if next_token == end_token_id:
                            break
                        
                        generated.append(next_token)
                        current_token = next_token
            
            generated_text = tokenizer.decode(generated)
            print(f"Generated: {generated_text}")
            print(f"{'='*60}\n")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                
                # Save encoder and decoder separately
                encoder_checkpoint = {
                    'model_state_dict': encoder.state_dict(),
                    'config': encoder_config,
                    'iter': it,
                    'val_loss': val_loss
                }
                decoder_checkpoint = {
                    'model_state_dict': decoder.state_dict(),
                    'config': decoder_config,
                    'iter': it,
                    'val_loss': val_loss
                }
                torch.save(encoder_checkpoint, os.path.join(args.out_dir, 'encoder.pt'))
                torch.save(decoder_checkpoint, os.path.join(args.out_dir, 'decoder.pt'))
                print(f"✓ Saved best model (val_loss: {val_loss:.4f})\n")
            
            encoder.train()
            decoder.train()
        
        # Save checkpoint
        if it > 0 and it % save_every == 0:
            checkpoint_dir = os.path.join(args.out_dir, f'checkpoint_{it:06d}')
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save encoder and decoder separately
            encoder_checkpoint = {
                'model_state_dict': encoder.state_dict(),
                'config': encoder_config,
                'iter': it
            }
            decoder_checkpoint = {
                'model_state_dict': decoder.state_dict(),
                'config': decoder_config,
                'iter': it
            }
            torch.save(encoder_checkpoint, os.path.join(checkpoint_dir, 'encoder.pt'))
            torch.save(decoder_checkpoint, os.path.join(checkpoint_dir, 'decoder.pt'))
            print(f"✓ Saved checkpoint at iter {it}\n")
        
        # Clear cache periodically
        if it % 100 == 0:
            torch.cuda.empty_cache()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
