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

from architecture.encoder import BidirectionalEncoder, create_encoder_config_from_bert, load_bert_encoder
from architecture.decoder import Transformer, load_decoder
from architecture.tokenizer import get_encoder_tokenizer, get_decoder_tokenizer
from dataloader.msmarco_loader import load_and_prepare_data as load_msmarco
from dataloader.tinystories_loader import load_and_prepare_data as load_tinystories


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


def validate_model(encoder, decoder, val_examples, tokenizer, device, dtype, max_dec_length, vocab_size, num_val=10):
    """Run validation and return loss."""
    encoder.eval()
    decoder.eval()
    
    val_loss = 0.0
    num_val = min(num_val, len(val_examples))
    
    with torch.no_grad():
        for val_idx in range(num_val):
            # CRITICAL: Reset context for each validation example
            
            example = val_examples[val_idx]
            
            context = example['context']
            question = example['question']
            answer = example['answer']
            
            # Clean question: strip all leading non-alphabetic characters
            while question and not question[0].isalpha():
                question = question[1:]
            
            # Clean answer: strip all leading non-alphabetic characters
            while answer and not answer[0].isalpha():
                answer = answer[1:]
            
            # Tokenize context and question with GPT tokenizer
            context_tokens = tokenizer.encode(context)
            question_tokens = tokenizer.encode(question)
            
            # Tokenize answer with GPT tokenizer
            answer_tokens = tokenizer.encode(answer)
            
            # Concatenate question + answer (natural GPT-OSS format)
            end_token_id = tokenizer.encode("<|endoftext|>", allowed_special={'<|endoftext|>'})[0]
            sequence_tokens = question_tokens + answer_tokens + [end_token_id]
            
            # Handle decoder overflow: move excess tokens to encoder context
            if len(sequence_tokens) > max_dec_length:
                overflow_tokens = sequence_tokens[:len(sequence_tokens) - max_dec_length]
                sequence_tokens = sequence_tokens[-max_dec_length:]
                # Append overflow to context
                context_tokens = context_tokens + overflow_tokens
            
            # Create separate tensors
            context_input = torch.tensor(context_tokens, dtype=torch.long, device=device)
            question_input = torch.tensor(question_tokens, dtype=torch.long, device=device)
            decoder_input = torch.tensor(sequence_tokens[:-1], dtype=torch.long, device=device)
            decoder_target = torch.tensor(sequence_tokens[1:], dtype=torch.long, device=device)
            
            with torch.amp.autocast(device_type='cuda', dtype=dtype):
                # Encode only context (question is in decoder)
                encoder_output = encoder(context_input, return_hidden_states=True)
                # Add batch dimension if needed
                if encoder_output.dim() == 2:
                    encoder_output = encoder_output.unsqueeze(0)
                logits, aux = decoder(decoder_input.unsqueeze(0), encoder_output=encoder_output)
                logits = logits.squeeze(0)
                loss = F.cross_entropy(logits.view(-1, vocab_size), decoder_target.view(-1))
            
            val_loss += loss.item()
    
    val_loss /= num_val
    
    encoder.train()
    decoder.train()
    
    return val_loss


def generate_samples(encoder, decoder, val_examples, tokenizer, device, dtype, num_samples=10):
    """Generate answer samples for validation examples (works for both MS MARCO and TinyStories)."""
    encoder.eval()
    decoder.eval()
    
    num_samples = min(num_samples, len(val_examples))
    samples = []
    
    print(f"\n{'='*60}")
    print(f"GENERATION DEBUG (first sample)")
    print(f"{'='*60}")
    
    for val_idx in range(num_samples):
        example = val_examples[val_idx]
        
        # Prepare question (strip all leading non-alphabetic characters)
        # For TinyStories, question is empty
        question = example['question']
        while question and not question[0].isalpha():
            question = question[1:]
        
        # Determine dataset type for debugging
        is_story = len(question) == 0
        dataset_type = "TinyStories" if is_story else "MS MARCO"
        
        # Debug first sample
        if val_idx == 0:
            print(f"Dataset type: {dataset_type}")
            print(f"Question: {question[:100] if question else '(empty - story continuation)'}")
            print(f"Context: {example['context'][:150]}...")
        
        # Tokenize context and question with GPT tokenizer
        context_tokens = tokenizer.encode(example['context'])
        question_tokens = tokenizer.encode(question) if question else []
        context_input = torch.tensor(context_tokens, dtype=torch.long, device=device)
        
        if val_idx == 0:
            print(f"Context tokens: {len(context_tokens)}")
            print(f"Question tokens: {len(question_tokens)}")
        
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=dtype):
                # Encode only context (question is in decoder)
                encoder_kv = encoder(context_input, return_hidden_states=True)
                
                if val_idx == 0:
                    print(f"Encoder output shape: {encoder_kv.shape}")
                
                # Generate token by token using GPT tokenizer
                end_token_id = tokenizer.encode("<|endoftext|>", allowed_special={'<|endoftext|>'})[0]
                
                # Initialize generated tokens list
                generated = []
                
                # Feed ALL question tokens at once to set context (if any)
                if len(question_tokens) > 0:
                    question_input_tensor = torch.tensor([question_tokens], dtype=torch.long, device=device)
                    # Add batch dimension to encoder output if needed
                    enc_out = encoder_kv.unsqueeze(0) if encoder_kv.dim() == 2 else encoder_kv
                    logits, aux = decoder(question_input_tensor, encoder_output=enc_out)
                    # Start generating from last question token's prediction
                    current_token = torch.argmax(logits[0, -1, :], dim=-1).item()
                    
                    if val_idx == 0:
                        print(f"After question, predicting first token: {tokenizer.decode([current_token])}")
                else:
                    # For stories, start generation from a special start token or first predicted token
                    # Use encoder output to predict first token
                    enc_out = encoder_kv.unsqueeze(0) if encoder_kv.dim() == 2 else encoder_kv
                    # Feed encoder output with empty decoder input to get first token
                    # Use a dummy start token (could be BOS if available, or generate from nothing)
                    start_input = torch.tensor([[end_token_id]], dtype=torch.long, device=device)
                    logits, aux = decoder(start_input, encoder_output=enc_out)
                    current_token = torch.argmax(logits[0, -1, :]).item()
                    
                    if val_idx == 0:
                        print(f"Story continuation, predicting first token: {tokenizer.decode([current_token])}")
                
                max_gen_len = 64
                
                for step in range(max_gen_len):
                    if current_token == end_token_id:
                        if val_idx == 0:
                            print(f"Hit EOS at step {step}")
                        break
                    
                    generated.append(current_token)
                    
                    # Generate next token
                    token_input = torch.tensor([[current_token]], dtype=torch.long, device=device)
                    
                    # Add batch dimension to encoder output if needed
                    enc_out = encoder_kv.unsqueeze(0) if encoder_kv.dim() == 2 else encoder_kv
                    logits, aux = decoder(token_input, encoder_output=enc_out)
                    
                    # Use greedy decoding (argmax) for deterministic validation
                    current_token = torch.argmax(logits[0, -1, :]).item()
                
                # Decode generated tokens to text
                generated_text = tokenizer.decode(generated)
                
                if val_idx == 0:
                    print(f"Generated tokens: {len(generated)}")
                    print(f"Generated text: {generated_text}")
            print(f"{'='*60}\n")
        
        samples.append({
            'context': example['context'][:150],
            'question': question[:100] if question else '(story)',
            'ground_truth': example['answer'],
            'generated': generated_text,
            'type': dataset_type
        })
    
    encoder.train()
    decoder.train()
    
    return samples


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
    max_qa_len = 128
    max_dec_length = 128  # Decoder chunk size: matches encoder max length for efficient processing
    max_decoder_tokens = 2048  # Max tokens for decoder input; overflow goes to encoder context
    max_iters = 50000
    eval_interval = 100
    save_every = 500
    log_interval = 100
    
    encoder_lr = 1e-5
    decoder_lr = 1e-6        # Keep low - only training cross-attention layers
    cross_attn_lr = 3e-4     # Higher for new QA-specific layers
    warmup_iters = 500       # Longer warmup for stability
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
    
    # Get tokenizer - using GPT BPE tokenizer for BOTH encoder and decoder
    tokenizer = get_decoder_tokenizer()  # GPT/Tiktoken tokenizer
    
    vocab_size = tokenizer.n_vocab  # Tiktoken vocab: ~200K
    
    print(f"Vocab size (GPT BPE): {vocab_size}")
    print("Using GPT BPE tokenizer for both encoder and decoder\n")
    
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
    
        # Override encoder vocab to match GPT tokenizer
        encoder_config.vocab_size = vocab_size
        print(f"  Vocab size (overridden to GPT): {encoder_config.vocab_size}")
        print(f"  Hidden size: {encoder_config.hidden_size}")
        print(f"  Layers: {encoder_config.num_hidden_layers}")
        print(f"  Attention heads: {encoder_config.num_attention_heads}")
        print(f"  Intermediate size: {encoder_config.intermediate_size}")
        print(f"  Max position: {encoder_config.max_position_embeddings}\n")
        
        # ========================================
        # STEP 2: Create encoder from scratch
        # ========================================
        print("="*60)
        print("STEP 2: Creating encoder from scratch")
        print("="*60)
        
        # Create encoder with random weights (training from scratch)
        encoder = BidirectionalEncoder(encoder_config, device=device)
        print("✓ Created encoder with random initialization")
        
        # Train all encoder parameters
        for name, param in encoder.named_parameters():
            param.requires_grad = True
        
        encoder_trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        encoder_total = sum(p.numel() for p in encoder.parameters())
        print(f"\n✓ Encoder ready: {encoder_trainable/1e6:.1f}M trainable / {encoder_total/1e6:.1f}M total")
        print(f"  Training: ALL encoder layers (from scratch)\n")
        
        # ========================================
        # STEP 3: Create decoder
        # ========================================
        print("="*60)
        print("STEP 3: Creating decoder")
        print("="*60)
        
        # Enable cross-attention with BERT encoder (1024 hidden size)
        encoder_hidden_size = encoder_config.hidden_size  # 1024 for BERT-large
        decoder = load_decoder(
            vocab_size=vocab_size,
            device=device,
            encoder_hidden_size=encoder_hidden_size,
        )
        
        # Train entire decoder from scratch
        for name, param in decoder.named_parameters():
            param.requires_grad = True
        
        trainable_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in decoder.parameters())
        print(f"\nDecoder: {trainable_params/1e6:.1f}M trainable / {total_params/1e6:.1f}M total")
        print(f"  Training: ALL decoder layers (from scratch)\n")
    
    # ========================================
    # Training setup
    # ========================================
    print("="*60)
    print("TRAINING SETUP")
    print("="*60)
    print("Architecture: Context encoder → decoder with question + answer")
    print("  - Context encoder output: [c_len, 1024]")
    print("  - Decoder input: [question tokens] [answer tokens]")
    print("  - Decoder cross-attention: attends to context encoder K,V")
    print(f"  - Decoder chunk size: {max_dec_length} (1=token-by-token, >1=chunked)\n")
    
    # Setup optimizer with encoder embeddings and decoder cross-attention parameters
    trainable_params = [
        *[p for p in encoder.parameters() if p.requires_grad],
        *[p for p in decoder.parameters() if p.requires_grad]
    ]
    
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cross_attn_lr,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )
    
    print("✓ Optimizer created:")
    print(f"  Learning rate: {cross_attn_lr}")
    print(f"  Trainable params: {sum(p.numel() for p in trainable_params)/1e6:.1f}M")
    print(f"    - Encoder embeddings: {sum(p.numel() for p in encoder.parameters() if p.requires_grad)/1e6:.1f}M")
    print(f"    - Decoder cross-attn + context layers: {sum(p.numel() for p in decoder.parameters() if p.requires_grad)/1e6:.1f}M\n")
    
    # ========================================
    # SANITY CHECK: Test decoder fluency before training
    # ========================================
    print("="*60)
    print("SANITY CHECK: Testing GPT-OSS decoder fluency")
    print("="*60)
    
    decoder.eval()
    with torch.no_grad():
        # Test pure decoder generation (GPT-OSS native forward)
        test_prompt = "The capital of France is"
        test_tokens = tokenizer.encode(test_prompt)
        print(f"Test prompt: '{test_prompt}'")
        print(f"Tokens: {test_tokens[:10]}...")
        
        # Use batched input [batch, seq_len]
        input_ids = torch.tensor([test_tokens], dtype=torch.long, device=device)
        
        with torch.amp.autocast(device_type='cuda', dtype=dtype):
            # Generate tokens
            max_new_tokens = 30
            generated_tokens = test_tokens.copy()
            
            for _ in range(max_new_tokens):
                # GPT-OSS forward: returns (logits, aux_dict)
                logits, aux_dict = decoder(input_ids)
                
                # Get next token (greedy) - batched, so logits is [batch, seq_len, vocab_size]
                next_token = torch.argmax(logits[0, -1, :]).item()
                
                # Check for EOS
                end_token_id = tokenizer.encode("<|endoftext|>", allowed_special={'<|endoftext|>'})[0]
                if next_token == end_token_id:
                    break
                
                generated_tokens.append(next_token)
                
                # Update input - append to batched sequence
                input_ids = torch.cat([
                    input_ids,
                    torch.tensor([[next_token]], dtype=torch.long, device=device)
                ], dim=1)
            
            full_text = tokenizer.decode(generated_tokens)
            print(f"Generated: '{full_text}'")
            
            # Check if output is reasonable
            if len(full_text) > len(test_prompt) + 10:
                print("✅ Decoder generated text!")
                if any(c.isalpha() for c in full_text[len(test_prompt):]):
                    print("✅ Generated text contains letters - GPT-OSS is working!")
                else:
                    print("⚠️  Generated text has no letters")
            else:
                print("⚠️  Decoder generated very little text")
    decoder.train()
    
    # Load datasets
    print("="*60)
    print("LOADING DATASETS")
    print("="*60)
    
    # Load MS MARCO
    print("\nLoading MS MARCO...")
    msmarco_dataset = load_dataset("ms_marco", "v2.1")
    msmarco_train, msmarco_val = load_msmarco(msmarco_dataset, tokenizer)
    
    # Load TinyStories
    print("\nLoading TinyStories...")
    tinystories_dataset = load_dataset("roneneldan/TinyStories")
    tinystories_train, tinystories_val = load_tinystories(tinystories_dataset, tokenizer, max_length=2048)
    
    # Process TinyStories to split into context/target for encoder-decoder training
    print("\nProcessing TinyStories for encoder-decoder training...")
    processed_tinystories_train = []
    for story_ex in tinystories_train:
        story = story_ex['text']
        # Split story roughly in half: first half = context, second half = answer
        story_tokens = tokenizer.encode(story)
        mid_point = len(story_tokens) // 2
        
        context_tokens = story_tokens[:mid_point]
        answer_tokens = story_tokens[mid_point:]
        
        # Decode back to text
        context = tokenizer.decode(context_tokens)
        answer = tokenizer.decode(answer_tokens)
        
        processed_tinystories_train.append({
            'context': context,
            'question': '',  # No question for stories
            'answer': answer
        })
    
    processed_tinystories_val = []
    for story_ex in tinystories_val:
        story = story_ex['text']
        story_tokens = tokenizer.encode(story)
        mid_point = len(story_tokens) // 2
        
        context_tokens = story_tokens[:mid_point]
        answer_tokens = story_tokens[mid_point:]
        
        context = tokenizer.decode(context_tokens)
        answer = tokenizer.decode(answer_tokens)
        
        processed_tinystories_val.append({
            'context': context,
            'question': '',
            'answer': answer
        })
    
    # Combine datasets
    train_examples = msmarco_train + processed_tinystories_train
    val_examples = msmarco_val + processed_tinystories_val
    
    print(f"\n✓ Total training examples: {len(train_examples)}")
    print(f"  - MS MARCO: {len(msmarco_train)}")
    print(f"  - TinyStories: {len(processed_tinystories_train)}")
    print(f"\n✓ Total validation examples: {len(val_examples)}")
    print(f"  - MS MARCO: {len(msmarco_val)}")
    print(f"  - TinyStories: {len(processed_tinystories_val)}\n")
    
    best_val_loss = float('inf')
    train_idx = 0
    
    for it in range(max_iters):
        # Get learning rate for cross-attention (only trainable params)
        cross_lr = get_lr(it, warmup_iters, max_iters, cross_attn_lr, cross_attn_lr * min_lr_ratio)
        
        optimizer.param_groups[0]['lr'] = cross_lr
        
        # Gradient accumulation
        total_loss = 0.0
        for micro_step in range(gradient_accumulation_steps):
            # CRITICAL: Reset context before each example to prevent bleeding
            
            # Get batch
            example = train_examples[train_idx % len(train_examples)]
            train_idx += 1
            
            context = example['context']
            question = example['question']
            answer = example['answer']
            
            # Clean question: strip all leading non-alphabetic characters
            while question and not question[0].isalpha():
                question = question[1:]
            
            # Clean answer: strip all leading non-alphabetic characters
            while answer and not answer[0].isalpha():
                answer = answer[1:]
            
            # Tokenize context and question with GPT tokenizer
            context_tokens = tokenizer.encode(context)
            question_tokens = tokenizer.encode(question)
            
            # Skip this example if question is empty after cleaning/tokenization
            if len(question_tokens) == 0:
                continue
            
            # Tokenize answer with GPT tokenizer
            answer_tokens = tokenizer.encode(answer)
            
            # Skip if answer is empty
            if len(answer_tokens) == 0:
                continue
            
            # Concatenate question + answer (natural GPT-OSS format)
            # Format: [question tokens] [answer tokens] [<|endoftext|>]
            end_token_id = tokenizer.encode("<|endoftext|>", allowed_special={'<|endoftext|>'})[0]
            sequence_tokens = question_tokens + answer_tokens + [end_token_id]
            
            # Handle decoder overflow: move excess tokens to encoder context
            if len(sequence_tokens) > max_decoder_tokens:
                overflow_tokens = sequence_tokens[:len(sequence_tokens) - max_decoder_tokens]
                sequence_tokens = sequence_tokens[-max_decoder_tokens:]
                # Append overflow to context
                context_tokens = context_tokens + overflow_tokens
            
            # Create separate tensors for context and question (for encoder)
            context_input = torch.tensor(context_tokens, dtype=torch.long, device=device)
            question_input = torch.tensor(question_tokens, dtype=torch.long, device=device)
            
            # Create decoder input/target from concatenated sequence
            # Input: question + answer[:-1]  Target: question[1:] + answer + EOS
            decoder_input = torch.tensor(sequence_tokens[:-1], dtype=torch.long, device=device)
            decoder_target = torch.tensor(sequence_tokens[1:], dtype=torch.long, device=device)
            
            # Forward pass
            with torch.amp.autocast(device_type='cuda', dtype=dtype):
                # Encode only the context (question is already in decoder input)
                if context_input.numel() > 0:
                    encoder_output = encoder(context_input, return_hidden_states=True)
                    # Add batch dimension if needed
                    if encoder_output.dim() == 2:
                        encoder_output = encoder_output.unsqueeze(0)
                else:
                    encoder_output = None
                
                # STEP 4: Decode
                logits, aux = decoder(
                    decoder_input.unsqueeze(0),
                    encoder_output=encoder_output
                )
                logits = logits.squeeze(0)
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
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        # Validation
        if it > 0 and it % eval_interval == 0:
            # Run validation
            val_loss = validate_model(
                encoder, decoder, val_examples, tokenizer, 
                device, dtype, max_dec_length, vocab_size, num_val=10
            )
            
            print(f"\n{'='*60}")
            print(f"VALIDATION @ Iter {it}")
            print(f"{'='*60}")
            print(f"Val Loss: {val_loss:.4f}")
            
            # Generate samples
            samples = generate_samples(
                encoder, decoder, val_examples, tokenizer, 
                device, dtype, num_samples=10
            )
            
            print(f"\nSample Generations ({len(samples)} examples):")
            print(f"{'='*60}")
            
            for idx, sample in enumerate(samples):
                print(f"\n--- Example {idx + 1}/{len(samples)} [{sample['type']}] ---")
                print(f"Context: {sample['context']}...")
                print(f"Question: {sample['question']}")
                print(f"Ground Truth: {sample['ground_truth'][:100]}...")
                print(f"Generated: {sample['generated']}")
            
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
                
                # Save tokenizer (unified GPT tokenizer)
                # For tiktoken, just save a marker file since it's reconstructed by code
                with open(os.path.join(args.out_dir, 'tokenizer.txt'), 'w') as f:
                    f.write('Use get_decoder_tokenizer() to load this GPT tokenizer')
                
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
            
            # Save tokenizer (unified GPT tokenizer)
            with open(os.path.join(checkpoint_dir, 'tokenizer.txt'), 'w') as f:
                f.write('Use get_decoder_tokenizer() to load this GPT tokenizer')
            
            print(f"✓ Saved checkpoint at iter {it}\n")
        
        # Clear cache periodically
        if it % 100 == 0:
            torch.cuda.empty_cache()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
