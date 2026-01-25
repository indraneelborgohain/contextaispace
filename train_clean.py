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
from architecture.tokenizer import get_encoder_tokenizer, get_decoder_tokenizer


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
    embedding_loaded = False
    for emb_key in ['wte.weight', 'transformer.wte.weight', 'embedding.weight']:
        if emb_key in gptoss_state:
            gpt_emb = gptoss_state[emb_key]
            dec_emb = decoder.embedding.weight
            print(f"  GPT-OSS embedding shape: {gpt_emb.shape}")
            print(f"  Decoder embedding shape: {dec_emb.shape}")
            min_vocab = min(gpt_emb.size(0), dec_emb.size(0))
            if gpt_emb.size(1) == dec_emb.size(1):
                decoder.embedding.weight.data[:min_vocab] = gpt_emb[:min_vocab].to(device)
                loaded_count += 1
                embedding_loaded = True
                print(f"✓ Loaded embeddings ({min_vocab} / {dec_emb.size(0)} tokens)")
                if min_vocab < dec_emb.size(0) * 0.5:
                    print(f"⚠️  WARNING: Only loaded {min_vocab}/{dec_emb.size(0)} embeddings ({min_vocab/dec_emb.size(0)*100:.1f}%)")
                    print(f"⚠️  Most tokens will have RANDOM embeddings!")
                break
            else:
                print(f"⚠️  Hidden size mismatch: GPT-OSS {gpt_emb.size(1)} vs Decoder {dec_emb.size(1)}")
    
    if not embedding_loaded:
        print("⚠️  WARNING: No embeddings loaded from GPT-OSS!")
        print("⚠️  Decoder will use RANDOM embeddings!")
    
    # 2. Load transformer layers (attention only - MoE/FFN likely incompatible)
    num_layers = decoder_config.num_hidden_layers
    print(f"Loading {num_layers} transformer layers...")
    print(f"Note: Only loading attention layers, NOT MoE/FFN (architecture may differ)\n")
    
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
            
            print(f"  ✓ Layer {layer_idx} (attention only)")
            break
    
    print(f"\n✓ Successfully loaded {loaded_count} parameter groups from GPT-OSS")
    print(f"⚠️  MoE/FFN layers NOT loaded - will train from scratch")
    print(f"⚠️  This means only embeddings + attention are pretrained")
    print(f"Decoder total parameters: {sum(p.numel() for p in decoder.parameters())/1e6:.1f}M\n")
    
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


def load_and_prepare_data(dataset, tokenizer):
    """Load MS MARCO dataset and prepare training/validation examples."""
    print("Loading MS MARCO dataset...")
    
    # Prepare training examples
    train_examples = []
    for example in dataset['train']:
        if not example.get('passages') or not example['passages'].get('passage_text'):
            continue
        if not example.get('query'):
            continue
        if not example.get('answers') or len(example['answers']) == 0:
            continue
        
        answer = example['answers'][0]
        
        # Skip unanswerable questions (MS MARCO has many "No Answer Present." examples)
        if answer.strip().lower() in ['no answer present.', 'no answer present', 'no answer']:
            continue
        if len(answer.strip()) == 0:
            continue
        
        # Concatenate all context passages (not just the first one)
        context = ' '.join(example['passages']['passage_text'])
        question = example['query']
        
        train_examples.append({
            'context': context,
            'question': question,
            'answer': answer
        })
    
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
        
        answer = example['answers'][0]
        
        # Skip unanswerable questions (MS MARCO has many "No Answer Present." examples)
        if answer.strip().lower() in ['no answer present.', 'no answer present', 'no answer']:
            continue
        if len(answer.strip()) == 0:
            continue
        
        # Concatenate all context passages (not just the first one)
        context = ' '.join(example['passages']['passage_text'])
        question = example['query']
        
        val_examples.append({
            'context': context,
            'question': question,
            'answer': answer
        })
    
    print(f"✓ Loaded {len(val_examples)} validation examples\n")
    
    return train_examples, val_examples


def validate_model(encoder, decoder, val_examples, tokenizer, device, dtype, max_dec_length, vocab_size, num_val=10):
    """Run validation and return loss."""
    encoder.eval()
    decoder.eval()
    
    val_loss = 0.0
    num_val = min(num_val, len(val_examples))
    
    with torch.no_grad():
        for val_idx in range(num_val):
            # CRITICAL: Reset context for each validation example
            decoder.reset_context()
            
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
            
            # Create separate tensors
            context_input = torch.tensor(context_tokens, dtype=torch.long, device=device)
            question_input = torch.tensor(question_tokens, dtype=torch.long, device=device)
            decoder_input = torch.tensor(sequence_tokens[:-1], dtype=torch.long, device=device)
            decoder_target = torch.tensor(sequence_tokens[1:], dtype=torch.long, device=device)
            
            with torch.amp.autocast(device_type='cuda', dtype=dtype):
                # Encode only context (question is in decoder)
                encoder_kv = encoder(context_input, return_hidden_states=True)
                logits = decoder(decoder_input, encoder_k=encoder_kv, encoder_v=encoder_kv, max_dec_length=max_dec_length)
                loss = F.cross_entropy(logits.view(-1, vocab_size), decoder_target.view(-1))
            
            val_loss += loss.item()
    
    val_loss /= num_val
    
    encoder.train()
    decoder.train()
    
    return val_loss


def generate_samples(encoder, decoder, val_examples, tokenizer, device, dtype, num_samples=10):
    """Generate answer samples for validation examples."""
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
        question = example['question']
        while question and not question[0].isalpha():
            question = question[1:]
        
        # Debug first sample
        if val_idx == 0:
            print(f"Question: {question[:100]}")
            print(f"Context: {example['context'][:150]}...")
        
        # Tokenize context and question with GPT tokenizer
        context_tokens = tokenizer.encode(example['context'])
        question_tokens = tokenizer.encode(question)
        context_input = torch.tensor(context_tokens, dtype=torch.long, device=device)
        question_input = torch.tensor(question_tokens, dtype=torch.long, device=device)
        
        if val_idx == 0:
            print(f"Context tokens: {len(context_tokens)}")
            print(f"Question tokens: {len(question_tokens)}")
        
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=dtype):
                # Encode only context (question is in decoder)
                encoder_kv = encoder(context_input, return_hidden_states=True)
                
                if val_idx == 0:
                    print(f"Encoder output shape: {encoder_kv.shape}")
                
                # CRITICAL: Reset decoder context before generation
                decoder.reset_context()
                
                # Generate token by token using GPT tokenizer
                end_token_id = tokenizer.encode("<|endoftext|>", allowed_special={'<|endoftext|>'})[0]
                
                # Initialize generated tokens list
                generated = []
                
                # Feed ALL question tokens at once to set context
                if len(question_tokens) > 0:
                    question_input_tensor = torch.tensor(question_tokens, dtype=torch.long, device=device)
                    logits = decoder(
                        question_input_tensor,
                        encoder_k=encoder_kv,
                        encoder_v=encoder_kv,
                        update_context=True,
                        max_dec_length=len(question_tokens)
                    )
                    # Start generating from last question token's prediction
                    current_token = torch.argmax(logits[-1], dim=-1).item()
                    
                    if val_idx == 0:
                        print(f"After question, predicting first answer token: {tokenizer.decode([current_token])}")
                else:
                    current_token = end_token_id
                
                max_gen_len = 64
                
                for step in range(max_gen_len):
                    if current_token == end_token_id:
                        if val_idx == 0:
                            print(f"Hit EOS at step {step}")
                        break
                    
                    generated.append(current_token)
                    
                    # Generate next token
                    token_input = torch.tensor([current_token], dtype=torch.long, device=device)
                    
                    logits = decoder(
                        token_input,
                        encoder_k=encoder_kv,
                        encoder_v=encoder_kv,
                        update_context=True,
                        max_dec_length=1
                    )
                    
                    # Use greedy decoding (argmax) for deterministic validation
                    current_token = torch.argmax(logits[-1], dim=-1).item()
        
        generated_text = tokenizer.decode(generated) if generated else "[empty]"
        
        if val_idx == 0:
            print(f"Generated tokens: {len(generated)}")
            print(f"Generated text: {generated_text}")
            print(f"{'='*60}\n")
        
        samples.append({
            'context': example['context'][:150],
            'question': question[:100],
            'ground_truth': example['answer'],
            'generated': generated_text
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
    
    # Get tokenizer - using GPT tokenizer for BOTH encoder and decoder
    tokenizer = get_decoder_tokenizer()  # GPT/Tiktoken tokenizer
    
    vocab_size = tokenizer.n_vocab  # Tiktoken vocab: ~200000+
    
    print(f"Vocab size (GPT/Tiktoken): {vocab_size}")
    print("Using GPT tokenizer for both encoder and decoder\n")
    
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
        
        # Make encoder embeddings and top 3 layers trainable for semantic learning
        num_layers = encoder_config.num_hidden_layers
        trainable_layers = 3  # Train top 3 layers
        
        for name, param in encoder.named_parameters():
            # Always train embeddings (adapting to new tokenizer)
            if 'embedding' in name:
                param.requires_grad = True
                print(f"  ✓ Encoder trainable: {name}")
            # Train top N layers (semantic adaptation)
            elif any(f'layer.{i}.' in name for i in range(num_layers - trainable_layers, num_layers)):
                param.requires_grad = True
                if 'layer' in name and any(f'layer.{i}.' in name for i in range(num_layers - trainable_layers, num_layers)):
                    layer_num = int(name.split('layer.')[1].split('.')[0])
                    if layer_num >= num_layers - trainable_layers:
                        print(f"  ✓ Encoder trainable: {name}")
            else:
                param.requires_grad = False
        
        encoder_trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        encoder_total = sum(p.numel() for p in encoder.parameters())
        print(f"\n✓ Encoder ready: {encoder_trainable/1e6:.1f}M trainable / {encoder_total/1e6:.1f}M total")
        print(f"  Training: embeddings + top {trainable_layers} layers (semantic adaptation)\n")
        
        # ========================================
        # STEP 3: Create decoder from GPT-OSS
        # ========================================
        # Match GPT-OSS architecture EXACTLY to load pretrained weights
        # GPT-OSS config from open-gpt-oss/model.py:
        #   vocab_size: 201_088
        #   hidden_size: 2880
        #   num_hidden_layers: 12
        #   num_attention_heads: 64
        #   num_key_value_heads: 8 (GQA with 8 groups)
        #   intermediate_size: 2880
        #   num_experts: 32
        #   experts_per_token: 4
        decoder_config = ModelConfig(
            vocab_size=vocab_size,              # ~201k (tiktoken/GPT-OSS vocab)
            hidden_size=2880,                   # MUST match GPT-OSS!
            num_hidden_layers=12,               # GPT-OSS has 12 layers
            num_attention_heads=64,             # GPT-OSS has 64 heads
            num_key_value_heads=8,              # GPT-OSS uses GQA with 8 KV heads
            intermediate_size=2880,             # GPT-OSS FFN size
            num_experts=32,                     # GPT-OSS has 32 experts
            experts_per_token=4,                # GPT-OSS uses top-4 routing
            use_moe=True,                       # GPT-OSS uses MoE
            head_dim=64,                        # 2880 / 45 ≈ 64 per head
            use_encoder_decoder_cross_attention=True,
            encoder_hidden_size=encoder_config.hidden_size,
            use_context_embedding=True,         # Enable context state tracking
            sliding_window=512                  # Sliding window attention
        )
        
        print(f"Decoder config:")
        print(f"  Vocab size: {decoder_config.vocab_size}")
        print(f"  Hidden size: {decoder_config.hidden_size}")
        print(f"  Layers: {decoder_config.num_hidden_layers}")
        print(f"  Cross-attention: {decoder_config.use_encoder_decoder_cross_attention}")
        print(f"  Encoder hidden size: {decoder_config.encoder_hidden_size}")
        print(f"  Sliding window: {decoder_config.sliding_window}")
        print(f"  Initial context length: {decoder_config.initial_context_length}")
        
        decoder = load_gptoss_decoder(decoder_config, gptoss_weights_dir, device)
        
        # Strategy: 
        # FROZEN: Embeddings, attention layers (loaded from GPT-OSS pretrained)
        # TRAINABLE: MoE/FFN, cross-attention, context layers (not in GPT-OSS weights)
        for name, param in decoder.named_parameters():
            # Freeze embeddings to preserve GPT-OSS language fluency
            if 'embedding' in name or 'unembedding' in name:
                param.requires_grad = False
                print(f"  ❄️ Frozen (GPT-OSS pretrained): {name}")
            # Freeze self-attention layers (loaded from GPT-OSS)
            elif 'attn.qkv' in name or 'attn.out' in name:
                param.requires_grad = False
            # Train MoE/FFN layers (NOT loaded from GPT-OSS - random init)
            elif 'mlp' in name or 'expert' in name or 'gate' in name.lower():
                param.requires_grad = True
                if 'experts.0' in name or 'gate.weight' in name:  # Log only first expert
                    print(f"  ✓ Trainable (random init): {name}")
            # Train cross-attention layers (NEW for QA task)
            elif 'cross_attn' in name or 'context_proj' in name or 'angle' in name:
                param.requires_grad = True
                print(f"  ✓ Trainable (QA-specific): {name}")
            # Freeze layer norms (GPT-OSS)
            elif 'norm' in name or 'ln' in name:
                param.requires_grad = False
            else:
                # Default: check if it looks like something from GPT-OSS
                param.requires_grad = False
        
        trainable_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in decoder.parameters())
        print(f"\nDecoder: {trainable_params/1e6:.1f}M trainable / {total_params/1e6:.1f}M total")
        print(f"  Frozen: embeddings + self-attention (GPT-OSS pretrained)")
        print(f"  Training: MoE/FFN + cross-attention + context (QA-specific)\n")
    
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
    
    # Load dataset
    dataset = load_dataset("ms_marco", "v2.1")
    train_examples, val_examples = load_and_prepare_data(dataset, tokenizer)
    
    # ========================================
    # SANITY CHECK: Test decoder fluency before training
    # ========================================
    print("="*60)
    print("SANITY CHECK: Testing decoder fluency (should be coherent English)")
    print("="*60)
     
    decoder.eval()
    with torch.no_grad():
        # Test pure decoder generation (no encoder, no training)
        test_prompt = "The capital of France is"
        test_tokens = tokenizer.encode(test_prompt)
        print(f"Test prompt: '{test_prompt}'")
        print(f"Tokens: {test_tokens}")
        
        decoder.reset_context()
        prompt_tensor = torch.tensor(test_tokens, dtype=torch.long, device=device)
        
        with torch.amp.autocast(device_type='cuda', dtype=dtype):
            # Feed prompt
            logits = decoder(prompt_tensor, encoder_k=None, encoder_v=None, update_context=True, max_dec_length=len(test_tokens))
            next_token = torch.argmax(logits[-1], dim=-1).item()
            
            # Generate 20 tokens
            generated = [next_token]
            for _ in range(19):
                token_tensor = torch.tensor([generated[-1]], dtype=torch.long, device=device)
                logits = decoder(token_tensor, encoder_k=None, encoder_v=None, update_context=True, max_dec_length=1)
                next_token = torch.argmax(logits[-1], dim=-1).item()
                generated.append(next_token)
            
            full_text = test_prompt + tokenizer.decode(generated)
            print(f"Generated: '{full_text}'")
            
            # Check if output is reasonable
            if len(full_text) < len(test_prompt) + 5:
                print("⚠️  WARNING: Decoder is generating very short text!")
            if not any(c.isalpha() for c in tokenizer.decode(generated)):
                print("⚠️  WARNING: Decoder is not generating alphabetic characters!")
            else:
                print("✓ Decoder appears to be working")
    
    print("="*60 + "\n")
    
    encoder.train()
    decoder.train()
    
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
            decoder.reset_context()
            
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
                    encoder_kv = encoder(context_input, return_hidden_states=True)
                    encoder_k = encoder_kv
                    encoder_v = encoder_kv
                else:
                    encoder_k = None
                    encoder_v = None
                
                # STEP 4: Decode
                logits = decoder(
                    decoder_input,
                    encoder_k=encoder_k,
                    encoder_v=encoder_v,
                    update_context=True,  # Allow context updates during training
                    max_dec_length=max_dec_length
                )
                
                # Debug: Print predicted text every 100 iterations
                if it % 100 == 0 and micro_step == 0:
                    with torch.no_grad():
                        predicted_tokens = torch.argmax(logits, dim=-1).cpu().tolist()
                        predicted_text = tokenizer.decode(predicted_tokens)
                        target_text = tokenizer.decode(decoder_target.cpu().tolist())
                        print(f"\n[Iter {it}] Training sample:")
                        print(f"  Predicted: {predicted_text[:200]}")
                        print(f"  Target:    {target_text[:200]}\n")
                
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
                print(f"\n--- Example {idx + 1}/{len(samples)} ---")
                print(f"Context: {sample['context']}...")
                print(f"Question: {sample['question']}")
                print(f"Ground Truth: {sample['ground_truth']}")
                print(f"Generated Answer: {sample['generated']}")
            
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
