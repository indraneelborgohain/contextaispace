import torch

import time,os,gc
import wandb
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR

from tqdm.notebook import tqdm


def text_to_token_ids(text, tokenizer, allowed_special='all'):
    encoded = tokenizer.encode(text, allowed_special=allowed_special)
    encoded_tensor = torch.tensor(encoded)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    return tokenizer.decode(token_ids.tolist())

def generate_next_token(model, idx, temperature=0.8, top_k=50, encoder_output=None, return_dict=False):
    """Generate next token using temperature and top-k sampling.
    
    Args:
        model: The decoder model
        idx: Current sequence of token ids (tensor)
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k filtering (None = no filtering)
        encoder_output: Optional encoder output for cross-attention
        return_dict: Whether model returns (logits, aux_dict) or just logits
    
    Returns:
        next_token: The sampled next token id (int)
    """
    with torch.inference_mode():
        if return_dict:
            logits, _ = model(idx, encoder_output=encoder_output, return_dict=True)
        else:
            logits = model(idx, encoder_output=encoder_output, return_dict=False)
    
    # Get logits for last token
    logits = logits[-1, :] / temperature
    
    # Apply top-k filtering
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[[-1]]] = -float('Inf')
    
    # Sample from distribution
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1).item()
    
    return next_token

def compute_loss_encoder_decoder(encoder, decoder, input_batch, target_batch, device, dtype, vocab_size):
    """Compute loss for encoder-decoder model using calcc pattern.
    
    Args:
        encoder: The encoder model
        decoder: The decoder model  
        input_batch: List of (context_input, decoder_input) tuples
        target_batch: List of decoder_target tensors
        device: Device to use
        dtype: Data type for autocast
        vocab_size: Vocabulary size
    
    Returns:
        Average loss over the batch
    """
    total_loss = 0
    for i in range(len(input_batch)):
        context_input, decoder_input = input_batch[i]
        context_input = context_input.to(device, non_blocking=True)
        decoder_input = decoder_input.to(device, non_blocking=True)
        target = target_batch[i].to(device, non_blocking=True)
        
        with torch.amp.autocast(device_type='cuda', dtype=dtype):
            # Encode context
            if context_input.numel() > 0:
                encoder_output = encoder(context_input, return_hidden_states=True)
            else:
                encoder_output = None
            
            # Decode
            logits, _ = decoder(decoder_input, encoder_output=encoder_output, return_dict=True)
            
            # Compute loss
            loss = F.cross_entropy(logits.view(-1, vocab_size), target.view(-1))
        
        total_loss += loss
        del context_input, decoder_input, target, logits, loss
    
    clear_gpu_memory()
    return total_loss / len(input_batch)

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
    
    num_val = min(num_val, len(val_examples))
    
    # Prepare batches for calcc-style computation
    input_batch = []
    target_batch = []
    
    with torch.no_grad():
        for val_idx in range(num_val):
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
            context_tokens = text_to_token_ids(context, tokenizer).tolist()
            question_tokens = text_to_token_ids(question, tokenizer).tolist()
            
            # Tokenize answer with GPT tokenizer
            answer_tokens = text_to_token_ids(answer, tokenizer).tolist()
            
            # Concatenate question + answer (natural GPT-OSS format)
            end_token_id = text_to_token_ids("<|endoftext|>", tokenizer).tolist()[0]
            sequence_tokens = question_tokens + answer_tokens + [end_token_id]
            
            # Handle decoder overflow: move excess tokens to encoder context
            if len(sequence_tokens) > max_dec_length:
                overflow_tokens = sequence_tokens[:len(sequence_tokens) - max_dec_length]
                sequence_tokens = sequence_tokens[-max_dec_length:]
                # Append overflow to context
                context_tokens = context_tokens + overflow_tokens
            
            # Create separate tensors
            context_input = torch.tensor(context_tokens, dtype=torch.long)
            decoder_input = torch.tensor(sequence_tokens[:-1], dtype=torch.long)
            decoder_target = torch.tensor(sequence_tokens[1:], dtype=torch.long)
            
            # Add to batch
            input_batch.append((context_input, decoder_input))
            target_batch.append(decoder_target)
        
        # Compute loss using compute_loss_encoder_decoder
        val_loss = compute_loss_encoder_decoder(
            encoder, decoder, input_batch, target_batch, device, dtype, vocab_size
        ).item()
    
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
        context_tokens = text_to_token_ids(example['context'], tokenizer).tolist()
        question_tokens = text_to_token_ids(question, tokenizer).tolist() if question else []
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
                end_token_id = text_to_token_ids("<|endoftext|>", tokenizer).tolist()[0]
                
                # Initialize generated tokens list
                generated = []
                
                # Feed ALL question tokens at once to set context (if any)
                if len(question_tokens) > 0:
                    question_input_tensor = torch.tensor(question_tokens, dtype=torch.long, device=device)
                    logits, aux = decoder(question_input_tensor, encoder_output=encoder_kv, return_dict=True)
                    # Start generating from last question token's prediction
                    current_token = torch.argmax(logits[-1, :], dim=-1).item()
                    
                    if val_idx == 0:
                        print(f"After question, predicting first token: {token_ids_to_text(torch.tensor([current_token]), tokenizer)}")
                else:
                    # For stories, start generation from a special start token or first predicted token
                    # Use encoder output to predict first token
                    # Feed encoder output with empty decoder input to get first token
                    # Use a dummy start token (could be BOS if available, or generate from nothing)
                    start_input = torch.tensor([end_token_id], dtype=torch.long, device=device)
                    logits, aux = decoder(start_input, encoder_output=encoder_kv, return_dict=True)
                    current_token = torch.argmax(logits[-1, :]).item()
                    
                    if val_idx == 0:
                        print(f"Story continuation, predicting first token: {token_ids_to_text(torch.tensor([current_token]), tokenizer)}")
                
                max_gen_len = 64
                
                for step in range(max_gen_len):
                    if current_token == end_token_id:
                        if val_idx == 0:
                            print(f"Hit EOS at step {step}")
                        break
                    
                    generated.append(current_token)
                    
                    # Generate next token using temperature sampling
                    token_input = torch.tensor([current_token], dtype=torch.long, device=device)
                    current_token = generate_next_token(
                        decoder,
                        token_input,
                        temperature=0.0,  # Greedy (deterministic)
                        top_k=1,
                        encoder_output=encoder_kv,
                        return_dict=True
                    )
                
                # Decode generated tokens to text
                generated_text = token_ids_to_text(torch.tensor(generated), tokenizer)
                
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



def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def calcc(input_batch, target_batch, model,device):
    total_loss=0
    for i in range(len(input_batch)):
        inp = input_batch[i].to(device, non_blocking=True)   
        logits = model(inp)
        del inp
        tgt = target_batch[i].to(device, non_blocking=True)
        loss= torch.nn.functional.cross_entropy(logits,tgt)
        total_loss += loss
        del tgt, logits, loss
        
    clear_gpu_memory()  
    return total_loss/len(input_batch)


def calc_loss_batch(input_batch, target_batch, model, device):
    loss=calcc(input_batch,target_batch,model,device)
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
            del loss
            
        else:
            break
    clear_gpu_memory()
    return total_loss / num_batches

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


# def generate_text(model, prompt, max_tokens=100, temperature=0.8, top_k=50):
#     """Generate text from a prompt using trained model."""
#     device = "cuda:0" if torch.cuda.is_available() else "cpu"
#     model = model.to(device)
#     model.eval()
#     
#     # Tokenize input
#     
#     idx = text_to_token_ids(prompt,tokenizer).to(device)
#     # Generate
#     for _ in range(max_tokens):
#         idx_cond = idx[-context_len:]
#         with torch.inference_mode():
#             logits= model(idx_cond)
#         logits = logits[-1, :] / temperature
# 
#         if top_k is not None:
#             v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
#             logits[logits < v[[-1]]] = -float('Inf')
# 
#         probs = F.softmax(logits, dim=-1)
#         idx_next = torch.multinomial(probs, num_samples=1)
#         idx = torch.cat((idx, idx_next), dim=0)
# 
#     
#     
#        
#     
#     # Decode and return
#     result = token_ids_to_text(idx,tokenizer)
#     return result
# 
# 
# 
# 
# def train_model(model, train_loader, val_loader, optimizer,scheduler, device, num_epochs,
#                        eval_freq, eval_iter, start_context):
#     
#     train_losses, val_losses, track_tokens_seen = [], [], []
#     tokens_seen, global_step = 0, -1
#     best_val_loss = float("inf")
#     for epoch in range(num_epochs):
#         model.train()
#         artifact = wandb.Artifact("gptoss-model", type="model")
#         for input_batch,target_batch in tqdm(train_loader):
# 
#             loss= calc_loss_batch(input_batch,target_batch,model,device)
#             loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()
#             scheduler.step()
#             tokens_seen += input_batch.numel()
#             global_step += 1
# 
#             wandb.log({"train/step_loss": loss.item(), 
#                        "tokens_seen": tokens_seen, 
#                        "epoch": epoch,
#                        "lr": optimizer.param_groups[0]["lr"]
#                        }, step=global_step)
#             
# 
#             if global_step % eval_freq == 0: 
#                 train_loss, val_loss = evaluate_model(
#                     model, train_loader, val_loader, device, eval_iter)
#                 train_losses.append(train_loss)
#                 val_losses.append(val_loss)
#                 track_tokens_seen.append(tokens_seen)
#                 print(f"Ep {epoch+1} (Step {global_step:06d}): "
#                       f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
#                 
#                 wandb.log({"train/loss": train_loss,
#                            "val/loss": val_loss}, step=global_step)
# 
#                 # Save best model
#                 if val_loss < best_val_loss:
#                     best_val_loss = val_loss
#                     torch.save(model.state_dict(), "model/gotoss_best.pt")
#                     artifact = wandb.Artifact("gptoss-model", type="model")
#                     
#                     artifact.add_file("model/gotoss_best.pt")
#                     wandb.log_artifact(artifact)
#                     print(f"✅ Saved new best model with val_loss={val_loss:.3f}")
#         torch.save(model.state_dict(),"model/gotoss.pt")
#         artifact = wandb.Artifact("gptoss-model", type="model")
#         artifact.add_file("model/gotoss.pt")
#         wandb.log_artifact(artifact, aliases=["latest"])
# 
#         torch.save([train_losses,val_losses,tokens_seen],"model/losses.pt")
# 
#         txt=generate_text(model,start_context)
#         print(txt)
#         wandb.log({"generated_text": wandb.Html(txt)}, step=global_step)
#         clear_gpu_memory()
#         
#         
#     return train_losses, val_losses, track_tokens_seen
# 
# def trainer(model,train_loader,val_loader,device):
#     learning_rate = 3e-4        
#     max_iters = 5         
#     warmup_steps = 100    
#     min_lr = 3e-5           
#     eval_iters = 5
#     eval_freq=150          
#     
#     
#     torch.manual_seed(123)
#     
#     print(f"Using {device}")
#    
#     if os.path.exists('model/gptoss.pt'):
#         model.load_state_dict(torch.load('model/gptoss.pt'))
#     
#     optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
#     scheduler_warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
#     scheduler_decay = CosineAnnealingLR(optimizer, T_max=max_iters - warmup_steps, eta_min=min_lr)
#     scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_decay], milestones=[warmup_steps])
#     num_epochs=max_iters
# 
# 
#     wandb.init(
#     project="gptoss-VS-gpt2",
#     name="gptoss-model",     
#     group="model-comparison",  
#     config={
#         "learning_rate": learning_rate,
#         "max_iters": max_iters,
#         "warmup_steps": warmup_steps,
#         "min_lr": min_lr,
#         "eval_iters": eval_iters,
#         "eval_freq": eval_freq,
#         "device": device,
#         "model_type": "gpt-oss"   
#     }
# )
# 
#     start_time = time.time()
#     train_losses, val_losses, tokens_seen = train_model(
#         model, train_loader, val_loader, optimizer,scheduler, device,
#         num_epochs=num_epochs, eval_freq=eval_freq, eval_iter=eval_iters,
#         start_context="a fast driver named Tim went for",
#     )
# 
#     torch.save(model.state_dict(),"model/gotoss.pt")
#     end_time = time.time()
#     execution_time_minutes = (end_time - start_time) / 60
#     wandb.log({"training_time_min": execution_time_minutes})
#     wandb.finish()
#     print(f"Training completed in {execution_time_minutes:.2f} minutes.") 
#     return train_losses,val_losses,tokens_seen