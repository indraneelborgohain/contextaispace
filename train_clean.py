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

from architecture.encoder import BidirectionalEncoder, create_encoder_config_from_bert, load_bert_encoder
from architecture.decoder import Transformer
from architecture.model_loader import load_decoder
from architecture.tokenizer import get_tokenizer
from dataloader.data_loader_context import create_context_dataloaders
from trainer import calcc, clear_gpu_memory,generate_samples, get_lr, text_to_token_ids, token_ids_to_text, validate_model
from trainer import compute_loss_encoder_decoder,generate_next_token



def startTrainning():
    

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
    tokenizer = get_tokenizer()  # GPT/Tiktoken tokenizer
    
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
        test_tokens = text_to_token_ids(test_prompt, tokenizer).tolist()
        print(f"Test prompt: '{test_prompt}'")
        print(f"Tokens: {test_tokens[:10]}...")
        
        # Use unbatched input [seq_len]
        input_ids = torch.tensor(test_tokens, dtype=torch.long, device=device)
        
        with torch.amp.autocast(device_type='cuda', dtype=dtype):
            # Generate tokens
            max_new_tokens = 30
            generated_tokens = test_tokens.copy()
            end_token_id = text_to_token_ids("<|endoftext|>", tokenizer).tolist()[0]
            
            for _ in range(max_new_tokens):
                # Generate next token with temperature sampling
                next_token = generate_next_token(
                    decoder, 
                    input_ids, 
                    temperature=0.8, 
                    top_k=50,
                    return_dict=True
                )
                
                # Check for EOS
                if next_token == end_token_id:
                    break
                
                generated_tokens.append(next_token)
                
                # Update input - append to unbatched sequence
                input_ids = torch.cat([
                    input_ids,
                    torch.tensor([next_token], dtype=torch.long, device=device)
                ], dim=0)
            
            full_text = token_ids_to_text(torch.tensor(generated_tokens), tokenizer)
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
    
    # Load datasets using unified dataloader
    print("="*60)
    print("LOADING DATASETS")
    print("="*60)
    
    train_loader, val_loader = create_context_dataloaders(
        batch_size=1,  # We handle batching via gradient accumulation
        num_workers=0,
        shuffle_train=True,
        max_tinystories=10000,
        max_msmarco=5000
    )
    
    print(f"\n✓ Created training dataloader with {len(train_loader)} batches")
    print(f"✓ Created validation dataloader with {len(val_loader)} batches\n")
    
    best_val_loss = float('inf')
    
    # Create iterator for infinite training loop
    train_iter = iter(train_loader)
    def get_batch():
        """Get next batch from train_loader, resetting iterator if needed."""
        nonlocal train_iter
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        return batch[0]  # Return first example (batch_size=1)
    
    for it in range(max_iters):
        # Get learning rate for cross-attention (only trainable params)
        cross_lr = get_lr(it, warmup_iters, max_iters, cross_attn_lr, cross_attn_lr * min_lr_ratio)
        
        optimizer.param_groups[0]['lr'] = cross_lr
        
        # Gradient accumulation - collect batch
        input_batch = []
        target_batch = []
        
        for micro_step in range(gradient_accumulation_steps):
            # Get batch from dataloader
            example = get_batch()
            
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
            context_tokens = text_to_token_ids(context, tokenizer).tolist()
            question_tokens = text_to_token_ids(question, tokenizer).tolist()
            
            # Skip this example if question is empty after cleaning/tokenization
            if len(question_tokens) == 0:
                continue
            
            # Tokenize answer with GPT tokenizer
            answer_tokens = text_to_token_ids(answer, tokenizer).tolist()
            
            # Skip if answer is empty
            if len(answer_tokens) == 0:
                continue
            
            # Concatenate question + answer (natural GPT-OSS format)
            # Format: [question tokens] [answer tokens] [<|endoftext|>]
            end_token_id = text_to_token_ids("<|endoftext|>", tokenizer).tolist()[0]
            sequence_tokens = question_tokens + answer_tokens + [end_token_id]
            
            # Handle decoder overflow: move excess tokens to encoder context
            if len(sequence_tokens) > max_decoder_tokens:
                overflow_tokens = sequence_tokens[:len(sequence_tokens) - max_decoder_tokens]
                sequence_tokens = sequence_tokens[-max_decoder_tokens:]
                # Append overflow to context
                context_tokens = context_tokens + overflow_tokens
            
            # Create separate tensors (without device, will be moved in compute_loss_encoder_decoder)
            context_input = torch.tensor(context_tokens, dtype=torch.long)
            decoder_input = torch.tensor(sequence_tokens[:-1], dtype=torch.long)
            decoder_target = torch.tensor(sequence_tokens[1:], dtype=torch.long)
            
            # Add to batch
            input_batch.append((context_input, decoder_input))
            target_batch.append(decoder_target)
        
        # Compute loss for entire batch using compute_loss_encoder_decoder
        if len(input_batch) > 0:
            loss = compute_loss_encoder_decoder(
                encoder, decoder, input_batch, target_batch, device, dtype, vocab_size
            )
            # Backward
            loss.backward()
            total_loss = loss.item()
        else:
            total_loss = 0.0
            
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
                    f.write('Use get_tokenizer() to load this GPT tokenizer')
                
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
                f.write('Use get_tokenizer() to load this GPT tokenizer')
            
            print(f"✓ Saved checkpoint at iter {it}\n")
        
        # Clear cache periodically
        if it % 100 == 0:
            clear_gpu_memory()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
