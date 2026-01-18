#!/usr/bin/env python3
"""
Clean training script for encoder-decoder model.
- Loads encoder from BERT (default: bert-large-uncased, medium config)
- Loads decoder from GPT-OSS weights
- Trains on MS MARCO dataset
"""
import argparse
import os
import time
import torch
import torch.nn.functional as F
from datasets import load_dataset

from architecture.config import ModelConfig
from architecture.encoder import BidirectionalEncoder, EncoderConfig
from architecture.transformer import Transformer
from architecture.tokenizer import get_tokenizer


def load_bert_encoder(encoder_config, bert_model_name, device):
    """Load BERT weights into encoder."""
    try:
        from transformers import AutoModel
    except ImportError:
        print("⚠️  transformers not installed. Install with: pip install transformers")
        return None
    
    print(f"\nLoading BERT weights from: {bert_model_name}")
    bert_model = AutoModel.from_pretrained(bert_model_name)
    bert_state = bert_model.state_dict()
    
    encoder = BidirectionalEncoder(encoder_config, device=device)
    encoder_state = encoder.state_dict()
    loaded_count = 0
    
    # Load embeddings
    if 'embeddings.word_embeddings.weight' in bert_state:
        bert_emb = bert_state['embeddings.word_embeddings.weight']
        enc_emb = encoder_state['embedding.weight']
        if enc_emb.shape == bert_emb.shape:
            encoder_state['embedding.weight'] = bert_emb.to(device)
            loaded_count += 1
    
    # Load encoder layers
    num_bert_layers = sum(1 for k in bert_state.keys() if k.startswith('encoder.layer.') and '.attention.self.query.weight' in k)
    num_enc_layers = encoder_config.num_hidden_layers
    layers_to_load = min(num_bert_layers, num_enc_layers)
    
    for layer_idx in range(layers_to_load):
        bert_prefix = f'encoder.layer.{layer_idx}'
        enc_prefix = f'blocks.{layer_idx}'
        
        # Load attention QKV
        if f'{bert_prefix}.attention.self.query.weight' in bert_state:
            q_weight = bert_state[f'{bert_prefix}.attention.self.query.weight']
            k_weight = bert_state[f'{bert_prefix}.attention.self.key.weight']
            v_weight = bert_state[f'{bert_prefix}.attention.self.value.weight']
            q_bias = bert_state[f'{bert_prefix}.attention.self.query.bias']
            k_bias = bert_state[f'{bert_prefix}.attention.self.key.bias']
            v_bias = bert_state[f'{bert_prefix}.attention.self.value.bias']
            
            qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
            qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
            
            encoder_state[f'{enc_prefix}.attn.qkv.weight'] = qkv_weight.to(device)
            encoder_state[f'{enc_prefix}.attn.qkv.bias'] = qkv_bias.to(device)
            loaded_count += 2
        
        # Load attention output
        if f'{bert_prefix}.attention.output.dense.weight' in bert_state:
            encoder_state[f'{enc_prefix}.attn.out.weight'] = bert_state[f'{bert_prefix}.attention.output.dense.weight'].to(device)
            encoder_state[f'{enc_prefix}.attn.out.bias'] = bert_state[f'{bert_prefix}.attention.output.dense.bias'].to(device)
            loaded_count += 2
        
        # Load FFN
        if f'{bert_prefix}.intermediate.dense.weight' in bert_state:
            encoder_state[f'{enc_prefix}.mlp.fc1.weight'] = bert_state[f'{bert_prefix}.intermediate.dense.weight'].to(device)
            encoder_state[f'{enc_prefix}.mlp.fc1.bias'] = bert_state[f'{bert_prefix}.intermediate.dense.bias'].to(device)
            loaded_count += 2
        
        if f'{bert_prefix}.output.dense.weight' in bert_state:
            encoder_state[f'{enc_prefix}.mlp.fc2.weight'] = bert_state[f'{bert_prefix}.output.dense.weight'].to(device)
            encoder_state[f'{enc_prefix}.mlp.fc2.bias'] = bert_state[f'{bert_prefix}.output.dense.bias'].to(device)
            loaded_count += 2
    
    # Load state into encoder
    for name, param in encoder.named_parameters():
        if name in encoder_state:
            param.data.copy_(encoder_state[name])
    for name, buf in encoder.named_buffers():
        if name in encoder_state:
            buf.data.copy_(encoder_state[name])
    
    print(f"✓ Loaded {loaded_count} parameters from BERT")
    return encoder


def load_gptoss_decoder(decoder_config, gptoss_weights_dir, device, max_layers=12):
    """Load GPT-OSS weights into decoder."""
    import glob
    
    print(f"\nLoading GPT-OSS weights from: {gptoss_weights_dir}")
    
    # Find safetensors files
    safetensor_files = glob.glob(os.path.join(gptoss_weights_dir, "*.safetensors"))
    
    if not safetensor_files:
        print("⚠️  No .safetensors files found, trying .pt files...")
        pt_files = glob.glob(os.path.join(gptoss_weights_dir, "*.pt"))
        if not pt_files:
            print("❌ No GPT-OSS weight files found")
            return None
    
    # Load GPT-OSS state
    try:
        from safetensors.torch import load_file
        gptoss_state = {}
        for f in safetensor_files:
            gptoss_state.update(load_file(f))
        print(f"✓ Loaded {len(safetensor_files)} safetensors files")
    except ImportError:
        print("⚠️  safetensors not installed, trying torch.load...")
        gptoss_state = torch.load(pt_files[0], map_location='cpu')
    
    decoder = Transformer(decoder_config, device=device)
    decoder_state = decoder.state_dict()
    loaded_count = 0
    
    # Load embedding
    for emb_key in ['wte.weight', 'transformer.wte.weight', 'embedding.weight']:
        if emb_key in gptoss_state:
            gpt_emb = gptoss_state[emb_key]
            dec_emb = decoder_state['embedding.weight']
            min_vocab = min(gpt_emb.size(0), dec_emb.size(0))
            if gpt_emb.size(1) == dec_emb.size(1):
                decoder_state['embedding.weight'][:min_vocab] = gpt_emb[:min_vocab].to(device)
                loaded_count += 1
                break
    
    # Load layers (limit to max_layers)
    layers_to_load = min(max_layers, decoder_config.num_hidden_layers)
    print(f"Loading first {layers_to_load} layers from GPT-OSS...")
    
    for layer_idx in range(layers_to_load):
        dec_prefix = f'block.{layer_idx}'
        
        # Try different GPT-OSS naming conventions
        for gpt_prefix in [f'layers.{layer_idx}', f'transformer.h.{layer_idx}', f'h.{layer_idx}']:
            # Load layer norm 1
            if f'{gpt_prefix}.ln_1.weight' in gptoss_state:
                decoder_state[f'{dec_prefix}.ln1.weight'] = gptoss_state[f'{gpt_prefix}.ln_1.weight'].to(device)
                decoder_state[f'{dec_prefix}.ln1.bias'] = gptoss_state[f'{gpt_prefix}.ln_1.bias'].to(device)
                loaded_count += 2
            
            # Load attention weights (could be c_attn or separate q,k,v)
            if f'{gpt_prefix}.attn.c_attn.weight' in gptoss_state:
                # Fused QKV
                decoder_state[f'{dec_prefix}.attn.qkv.weight'] = gptoss_state[f'{gpt_prefix}.attn.c_attn.weight'].to(device)
                decoder_state[f'{dec_prefix}.attn.qkv.bias'] = gptoss_state[f'{gpt_prefix}.attn.c_attn.bias'].to(device)
                loaded_count += 2
            
            # Load attention output
            if f'{gpt_prefix}.attn.c_proj.weight' in gptoss_state:
                decoder_state[f'{dec_prefix}.attn.out.weight'] = gptoss_state[f'{gpt_prefix}.attn.c_proj.weight'].to(device)
                decoder_state[f'{dec_prefix}.attn.out.bias'] = gptoss_state[f'{gpt_prefix}.attn.c_proj.bias'].to(device)
                loaded_count += 2
            
            if loaded_count > 0:
                break
    
    # Load state into decoder
    for name, param in decoder.named_parameters():
        if name in decoder_state:
            param.data.copy_(decoder_state[name])
    for name, buf in decoder.named_buffers():
        if name in decoder_state:
            buf.data.copy_(decoder_state[name])
    
    print(f"✓ Loaded {loaded_count} parameters from GPT-OSS")
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


def main():
    parser = argparse.ArgumentParser(description="Clean training script")
    parser.add_argument("--out_dir", type=str, default="model_clean", help="Output directory")
    args = parser.parse_args()
    
    # Fixed configuration (medium model)
    bert_model = "bert-large-uncased"
    gptoss_weights = "architecture/open-gpt-oss/weights"
    max_decoder_layers = 12
    
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
    
    print("="*60)
    print("Clean Training Script")
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
    
    # Load BERT model first to get actual config
    print("Loading BERT model to extract config...")
    try:
        from transformers import AutoModel
        bert_model = AutoModel.from_pretrained(bert_model)
        bert_config = bert_model.config
        
        # Create encoder config from BERT's actual config
        encoder_config = EncoderConfig(
            vocab_size=vocab_size,
            hidden_size=bert_config.hidden_size,
            num_hidden_layers=bert_config.num_hidden_layers,
            num_attention_heads=bert_config.num_attention_heads,
            num_key_value_heads=bert_config.num_attention_heads,  # BERT uses MHA
            intermediate_size=bert_config.intermediate_size
        )
        
        print(f"✓ Extracted config from BERT:")
        print(f"  Hidden size: {bert_config.hidden_size}")
        print(f"  Layers: {bert_config.num_hidden_layers}")
        print(f"  Attention heads: {bert_config.num_attention_heads}")
        print(f"  Intermediate size: {bert_config.intermediate_size}\n")
        
    except ImportError:
        print("⚠️  transformers not installed, using default medium config")
        encoder_config = EncoderConfig(
            vocab_size=vocab_size,
            hidden_size=1024,
            num_hidden_layers=12,
            num_attention_heads=16,
            num_key_value_heads=16
        )
        bert_model = None
    
    # Create decoder config (medium: GPT-OSS compatible)
    # Use encoder's actual hidden size for cross-attention compatibility
    decoder_config = ModelConfig(
        vocab_size=vocab_size,
        hidden_size=2880,
        num_hidden_layers=max_decoder_layers,
        num_experts=32,
        num_attention_heads=64,
        num_key_value_heads=64,
        use_encoder_decoder_cross_attention=True,
        encoder_hidden_size=encoder_config.hidden_size  # Use actual encoder hidden size
    )
    
    print(f"Encoder config: {encoder_config.hidden_size}d, {encoder_config.num_hidden_layers} layers")
    print(f"Decoder config: {decoder_config.hidden_size}d, {decoder_config.num_hidden_layers} layers\n")
    
    # Load encoder from BERT (reuse loaded BERT model if available)
    if bert_model is not None:
        print("Loading BERT weights into encoder...")
        encoder = BidirectionalEncoder(encoder_config, device=device)
        encoder_state = encoder.state_dict()
        bert_state = bert_model.state_dict()
        loaded_count = 0
        
        # Load embeddings
        if 'embeddings.word_embeddings.weight' in bert_state:
            bert_emb = bert_state['embeddings.word_embeddings.weight']
            enc_emb = encoder_state['embedding.weight']
            if enc_emb.shape == bert_emb.shape:
                encoder_state['embedding.weight'] = bert_emb.to(device)
                loaded_count += 1
        
        # Load encoder layers
        for layer_idx in range(encoder_config.num_hidden_layers):
            bert_prefix = f'encoder.layer.{layer_idx}'
            enc_prefix = f'blocks.{layer_idx}'
            
            # Load attention QKV
            if f'{bert_prefix}.attention.self.query.weight' in bert_state:
                q_weight = bert_state[f'{bert_prefix}.attention.self.query.weight']
                k_weight = bert_state[f'{bert_prefix}.attention.self.key.weight']
                v_weight = bert_state[f'{bert_prefix}.attention.self.value.weight']
                q_bias = bert_state[f'{bert_prefix}.attention.self.query.bias']
                k_bias = bert_state[f'{bert_prefix}.attention.self.key.bias']
                v_bias = bert_state[f'{bert_prefix}.attention.self.value.bias']
                
                qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
                qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
                
                encoder_state[f'{enc_prefix}.attn.qkv.weight'] = qkv_weight.to(device)
                encoder_state[f'{enc_prefix}.attn.qkv.bias'] = qkv_bias.to(device)
                loaded_count += 2
            
            # Load attention output
            if f'{bert_prefix}.attention.output.dense.weight' in bert_state:
                encoder_state[f'{enc_prefix}.attn.out.weight'] = bert_state[f'{bert_prefix}.attention.output.dense.weight'].to(device)
                encoder_state[f'{enc_prefix}.attn.out.bias'] = bert_state[f'{bert_prefix}.attention.output.dense.bias'].to(device)
                loaded_count += 2
            
            # Load FFN
            if f'{bert_prefix}.intermediate.dense.weight' in bert_state:
                encoder_state[f'{enc_prefix}.mlp.fc1.weight'] = bert_state[f'{bert_prefix}.intermediate.dense.weight'].to(device)
                encoder_state[f'{enc_prefix}.mlp.fc1.bias'] = bert_state[f'{bert_prefix}.intermediate.dense.bias'].to(device)
                loaded_count += 2
            
            if f'{bert_prefix}.output.dense.weight' in bert_state:
                encoder_state[f'{enc_prefix}.mlp.fc2.weight'] = bert_state[f'{bert_prefix}.output.dense.weight'].to(device)
                encoder_state[f'{enc_prefix}.mlp.fc2.bias'] = bert_state[f'{bert_prefix}.output.dense.bias'].to(device)
                loaded_count += 2
        
        # Load state into encoder
        for name, param in encoder.named_parameters():
            if name in encoder_state:
                param.data.copy_(encoder_state[name])
        for name, buf in encoder.named_buffers():
            if name in encoder_state:
                buf.data.copy_(encoder_state[name])
        
        print(f"✓ Loaded {loaded_count} parameters from BERT")
        del bert_model, bert_state  # Free memory
    else:
        encoder = load_bert_encoder(encoder_config, bert_model, device)
        if encoder is None:
            print("Failed to load BERT, creating random encoder")
            encoder = BidirectionalEncoder(encoder_config, device=device)
    
    print(f"Encoder params: {sum(p.numel() for p in encoder.parameters())/1e6:.2f}M\n")
    
    # Load decoder from GPT-OSS
    decoder = load_gptoss_decoder(decoder_config, gptoss_weights, device, max_decoder_layers)
    if decoder is None:
        print("Failed to load GPT-OSS, creating random decoder")
        decoder = Transformer(decoder_config, device=device)
    
    print(f"Decoder params: {sum(p.numel() for p in decoder.parameters())/1e6:.2f}M\n")
    
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
    
    print("✓ Optimizer created with 3-tier learning rates")
    print(f"  Encoder: {encoder_lr}")
    print(f"  Decoder: {decoder_lr}")
    print(f"  Cross-attn: {cross_attn_lr}\n")
    
    # Load dataset
    print("Loading MS MARCO dataset...")
    dataset = load_dataset("ms_marco", "v2.1")
    
    # Prepare training examples
    train_examples = []
    for i, ex in enumerate(dataset['train']):
        if i >= 5000:  # Limit for faster training
            break
        
        if 'passages' in ex and 'query' in ex:
            passages = ex['passages']
            if isinstance(passages, dict) and 'passage_text' in passages:
                passage_texts = passages['passage_text'][:1]  # Use first passage
                context = passage_texts[0] if isinstance(passage_texts, list) and len(passage_texts) > 0 else ""
            else:
                continue
            
            question = ex['query']
            
            # Get answers
            if 'answers' in ex and isinstance(ex['answers'], list) and len(ex['answers']) > 0:
                answer = ex['answers'][0]
            elif 'wellFormedAnswers' in ex and isinstance(ex['wellFormedAnswers'], list) and len(ex['wellFormedAnswers']) > 0:
                answer = ex['wellFormedAnswers'][0]
            else:
                answer = context[:50]  # Fallback
            
            # Tokenize
            context_tokens = tokenizer.encode(context)[:max_context_len - 50]
            question_tokens = tokenizer.encode(question)[:50]
            answer_tokens = tokenizer.encode(answer)[:50]
            
            c_marker = tokenizer.encode("<C>")
            sep_marker = tokenizer.encode("<SEP>")
            a_marker = tokenizer.encode("<A>")
            
            # Encoder input: <C> context <SEP> question
            ctx_tokens = c_marker + context_tokens + sep_marker + question_tokens
            
            # Decoder input: question <A> answer
            qa_tokens = question_tokens + a_marker + answer_tokens
            a_position = len(question_tokens)
            
            train_examples.append((ctx_tokens, qa_tokens, a_position))
    
    print(f"✓ Prepared {len(train_examples)} training examples\n")
    
    # Prepare validation examples (small subset)
    val_examples = train_examples[:100]  # Use first 100 for validation
    print(f"✓ Validation examples: {len(val_examples)}\n")
    
    # Training loop
    print("="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    encoder.train()
    decoder.train()
    
    iter_num = 0
    accum_step = 0
    running_loss = 0.0
    log_count = 0
    best_val_loss = float('inf')
    
    t0 = time.time()
    
    while iter_num < args.max_iters:
        # Sample random batch
        import random
        batch_examples = random.sample(train_examples, batch_size)
        
        batch_loss = 0.0
        
        for ctx_tokens, qa_tokens, a_position in batch_examples:
            ctx_tensor = torch.tensor(ctx_tokens, dtype=torch.long, device=device)
            qa_tensor = torch.tensor(qa_tokens, dtype=torch.long, device=device)
            
            # Forward pass
            with torch.autocast(device_type='cuda' if 'cuda' in str(device) else 'cpu', dtype=dtype, enabled=(dtype != torch.float32)):
                # Encode
                encoder_k, encoder_v = encoder(ctx_tensor, return_encoder_kv=True)
                
                # Decode
                decoder.reset_context()
                logits = decoder(qa_tensor, encoder_k=encoder_k, encoder_v=encoder_v)
                
                # Loss (only on answer tokens)
                seq_len = len(logits) - 1
                loss_mask = torch.zeros(seq_len, device=device, dtype=torch.float)
                if 0 <= a_position < seq_len:
                    loss_mask[a_position:] = 1.0
                else:
                    loss_mask[:] = 1.0
                
                ce_loss = F.cross_entropy(
                    logits[:-1].view(-1, logits.size(-1)),
                    qa_tensor[1:].view(-1),
                    reduction='none'
                )
                
                masked_loss = ce_loss * loss_mask
                loss = masked_loss.sum() / (loss_mask.sum() + 1e-8)
            
            # Scale by gradient accumulation
            loss = loss / gradient_accumulation_steps
            batch_loss += loss.item()
            
            # Backward
            loss.backward()
        
        # Accumulation step
        accum_step += 1
        
        if accum_step >= gradient_accumulation_steps:
            # Update weights
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), 1.0)
            
            # Update LR
            enc_lr = get_lr(iter_num, warmup_iters, max_iters, encoder_lr, encoder_lr * min_lr_ratio)
            dec_lr = get_lr(iter_num, warmup_iters, max_iters, decoder_lr, decoder_lr * min_lr_ratio)
            cross_lr = get_lr(iter_num, warmup_iters, max_iters, cross_attn_lr, cross_attn_lr * min_lr_ratio)
            
            optimizer.param_groups[0]['lr'] = enc_lr
            optimizer.param_groups[1]['lr'] = dec_lr
            optimizer.param_groups[2]['lr'] = cross_lr
            
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            accum_step = 0
            torch.cuda.empty_cache()
        
        running_loss += batch_loss
        log_count += 1
        
        # Logging
        if (iter_num + 1) % log_interval == 0:
            avg_loss = running_loss / log_count
            t1 = time.time()
            dt = (t1 - t0) / log_interval
            print(f"iter {iter_num + 1:5d} | loss {avg_loss:.4f} | {dt*1000:.2f}ms/iter")
            running_loss = 0.0
            log_count = 0
            t0 = time.time()
        
        # Validation
        if (iter_num + 1) % eval_interval == 0:
            print("\n" + "="*60)
            print(f"Validation at iteration {iter_num + 1}")
            print("="*60)
            
            encoder.eval()
            decoder.eval()
            
            val_loss = 0.0
            val_count = 0
            
            # Compute validation loss on a few examples
            import random
            val_batch = random.sample(val_examples, min(10, len(val_examples)))
            
            with torch.no_grad():
                for ctx_tokens, qa_tokens, a_position in val_batch:
                    ctx_tensor = torch.tensor(ctx_tokens, dtype=torch.long, device=device)
                    qa_tensor = torch.tensor(qa_tokens, dtype=torch.long, device=device)
                    
                    with torch.autocast(device_type='cuda' if 'cuda' in str(device) else 'cpu', dtype=dtype, enabled=(dtype != torch.float32)):
                        encoder_k, encoder_v = encoder(ctx_tensor, return_encoder_kv=True)
                        decoder.reset_context()
                        logits = decoder(qa_tensor, encoder_k=encoder_k, encoder_v=encoder_v)
                        
                        seq_len = len(logits) - 1
                        loss_mask = torch.zeros(seq_len, device=device, dtype=torch.float)
                        if 0 <= a_position < seq_len:
                            loss_mask[a_position:] = 1.0
                        else:
                            loss_mask[:] = 1.0
                        
                        ce_loss = F.cross_entropy(
                            logits[:-1].view(-1, logits.size(-1)),
                            qa_tensor[1:].view(-1),
                            reduction='none'
                        )
                        
                        masked_loss = ce_loss * loss_mask
                        loss = masked_loss.sum() / (loss_mask.sum() + 1e-8)
                    
                    val_loss += loss.item()
                    val_count += 1
            
            avg_val_loss = val_loss / val_count if val_count > 0 else 0.0
            print(f"\nValidation Loss: {avg_val_loss:.4f}\n")
            
            # Generate answer for one example to show progress
            ctx_tokens, qa_tokens, a_position = random.choice(val_examples)
            
            # Decode tokens to text for display
            c_marker = tokenizer.encode("<C>")
            sep_marker = tokenizer.encode("<SEP>")
            a_marker = tokenizer.encode("<A>")
            
            # Extract context and question from ctx_tokens
            ctx_start = len(c_marker)
            try:
                sep_idx = ctx_tokens.index(sep_marker[0])
                context_only = ctx_tokens[ctx_start:sep_idx]
                question_only = ctx_tokens[sep_idx + len(sep_marker):]
            except:
                context_only = ctx_tokens[ctx_start:]
                question_only = []
            
            # Extract question and answer from qa_tokens
            try:
                a_idx = qa_tokens.index(a_marker[0])
                question_from_qa = qa_tokens[:a_idx]
                answer_ground_truth = qa_tokens[a_idx + len(a_marker):]
            except:
                question_from_qa = qa_tokens
                answer_ground_truth = []
            
            # Generate answer
            ctx_tensor = torch.tensor(ctx_tokens, dtype=torch.long, device=device)
            question_tokens = question_from_qa
            
            with torch.no_grad():
                encoder_k, encoder_v = encoder(ctx_tensor, return_encoder_kv=True)
                
                # Start with question + <A>
                current_tokens = question_tokens + a_marker
                generated_tokens = []
                
                for _ in range(50):  # Generate up to 50 tokens
                    qa_tensor = torch.tensor(current_tokens, dtype=torch.long, device=device)
                    decoder.reset_context()
                    
                    with torch.autocast(device_type='cuda' if 'cuda' in str(device) else 'cpu', dtype=dtype, enabled=(dtype != torch.float32)):
                        logits = decoder(qa_tensor, encoder_k=encoder_k, encoder_v=encoder_v)
                    
                    next_token = logits[-1].argmax().item()
                    
                    if next_token == 0 or next_token == 50256:  # End token
                        break
                    
                    generated_tokens.append(next_token)
                    current_tokens.append(next_token)
            
            # Decode to text
            context_text = tokenizer.decode(context_only)
            question_text = tokenizer.decode(question_tokens)
            answer_gt_text = tokenizer.decode(answer_ground_truth)
            answer_gen_text = tokenizer.decode(generated_tokens)
            
            print("Sample Generation:")
            print(f"Context: {context_text[:200]}...")
            print(f"Question: {question_text}")
            print(f"Ground Truth: {answer_gt_text}")
            print(f"Generated: {answer_gen_text}")
            print("="*60 + "\n")
            
            encoder.train()
            decoder.train()
            
            # Update best model if needed
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_path = os.path.join(args.out_dir, "best_model.pt")
                torch.save({
                    'encoder': encoder.state_dict(),
                    'decoder': decoder.state_dict(),
                    'encoder_config': encoder_config,
                    'decoder_config': decoder_config,
                    'iter': iter_num + 1,
                    'val_loss': avg_val_loss
                }, best_path)
                print(f"💾 Saved best model (val_loss={avg_val_loss:.4f})\n")
        
        # Save checkpoint
        if save_every > 0 and (iter_num + 1) % save_every == 0:
            checkpoint_path = os.path.join(args.out_dir, f"checkpoint_{iter_num + 1}.pt")
            torch.save({
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'encoder_config': encoder_config,
                'decoder_config': decoder_config,
                'optimizer': optimizer.state_dict(),
                'iter': iter_num + 1,
            }, checkpoint_path)
            print(f"💾 Saved checkpoint: {checkpoint_path}")
        
        iter_num += 1
    
    # Save final model
    final_path = os.path.join(args.out_dir, "final_model.pt")
    torch.save({
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'encoder_config': encoder_config,
        'decoder_config': decoder_config
    }, final_path)
    
    print(f"\n✓ Training complete! Saved to {final_path}")


if __name__ == "__main__":
    main()
