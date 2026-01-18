#!/usr/bin/env python3
"""
Simple script to test loading model checkpoint and doing inference.
This helps diagnose GPU memory issues.
"""
import argparse
import torch
from datasets import load_dataset

try:
    from accelerate import load_checkpoint_in_model, init_empty_weights
    ACCELERATE_AVAILABLE = True
except ImportError:
    print("⚠️  accelerate not installed. Install with: pip install accelerate")
    ACCELERATE_AVAILABLE = False

from architecture.config import ModelConfig
from architecture.encoder import BidirectionalEncoder, EncoderConfig
from architecture.transformer import Transformer
from architecture.tokenizer import get_tokenizer


def print_gpu_memory(label):
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        print(f"[{label}] GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")


def load_checkpoint_metadata_only(checkpoint_path):
    """Load only the config metadata without loading full state dicts"""
    # Use mmap if available (PyTorch 2.0+)
    try:
        checkpoint = torch.load(
            checkpoint_path, 
            map_location='cpu', 
            weights_only=False,
            mmap=True  # Memory-mapped loading
        )
    except TypeError:
        # Fallback for older PyTorch versions
        checkpoint = torch.load(
            checkpoint_path, 
            map_location='cpu', 
            weights_only=False
        )
    return checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--use-bfloat16", action="store_true", help="Use bfloat16 precision")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.use_bfloat16 else torch.float32
    print(f"Using device: {device}, dtype: {dtype}\n")
    
    # Get tokenizer
    tokenizer = get_tokenizer()
    
    # STEP 1: Load checkpoint metadata only (configs)
    print("=" * 60)
    print("STEP 1: Loading checkpoint metadata")
    print("=" * 60)
    print_gpu_memory("Before loading checkpoint")
    
    # Load checkpoint with memory mapping
    checkpoint = load_checkpoint_metadata_only(args.checkpoint)
    print(f"✓ Checkpoint loaded")
    print(f"  Keys in checkpoint: {list(checkpoint.keys())}")
    
    encoder_config = checkpoint.get('encoder_config')
    if isinstance(encoder_config, dict):
        encoder_config = EncoderConfig(**encoder_config)
    
    decoder_config = checkpoint.get('decoder_config')
    if isinstance(decoder_config, dict):
        decoder_config = ModelConfig(**decoder_config)
    
    print(f"Encoder config: {encoder_config.hidden_size}d, {encoder_config.num_hidden_layers} layers")
    print(f"Decoder config: {decoder_config.hidden_size}d, {decoder_config.num_hidden_layers} layers")
    
    # STEP 2: Create models and load weights efficiently
    print("\n" + "=" * 60)
    print("STEP 2: Creating and loading models")
    print("=" * 60)
    print_gpu_memory("Before creating models")
    
    if ACCELERATE_AVAILABLE:
        print("🚀 Using Accelerate for memory-efficient loading\n")
        
        # NOTE: Accelerate's load_checkpoint_in_model expects a file path, not state dict
        # So we'll use init_empty_weights + manual loading for better memory efficiency
        
        # Create encoder with empty weights (no memory allocated)
        print("Creating encoder with empty weights...")
        with init_empty_weights():
            encoder = BidirectionalEncoder(encoder_config, device='meta')
        print(f"  Encoder params: {sum(p.numel() for p in encoder.parameters())/1e6:.2f}M (meta device)")
        
        # Load encoder weights directly to target device/dtype
        if 'encoder' in checkpoint:
            print("Loading encoder weights to GPU...")
            # Load state dict directly with target device/dtype
            encoder_state = checkpoint['encoder']
            
            # Move model to device first (creates actual tensors)
            encoder = encoder.to_empty(device=device)
            
            # Load weights piece by piece to target device/dtype
            for name, param in encoder.named_parameters():
                if name in encoder_state:
                    param.data.copy_(encoder_state[name].to(device=device, dtype=dtype))
            
            for name, buffer in encoder.named_buffers():
                if name in encoder_state:
                    buffer.data.copy_(encoder_state[name].to(device=device, dtype=dtype))
            
            print(f"✓ Encoder loaded to {device} with {dtype}")
            print_gpu_memory("After loading encoder")
            
            # Free encoder weights from CPU memory
            del checkpoint['encoder']
            del encoder_state
            torch.cuda.empty_cache() 
        
        # Create decoder with empty weights
        print("\nCreating decoder with empty weights...")
        with init_empty_weights():
            decoder = Transformer(decoder_config, device='meta')
        print(f"  Decoder params: {sum(p.numel() for p in decoder.parameters())/1e6:.2f}M (meta device)")
        
        # Load decoder weights directly to target device/dtype
        if 'decoder' in checkpoint:
            print("Loading decoder weights to GPU...")
            decoder_state = checkpoint['decoder']
            
            # Move model to device first
            decoder = decoder.to_empty(device=device)
            
            # Load weights piece by piece to target device/dtype
            for name, param in decoder.named_parameters():
                if name in decoder_state:
                    param.data.copy_(decoder_state[name].to(device=device, dtype=dtype))
            
            for name, buffer in decoder.named_buffers():
                if name in decoder_state:
                    buffer.data.copy_(decoder_state[name].to(device=device, dtype=dtype))
            
            print(f"✓ Decoder loaded to {device} with {dtype}")
            print_gpu_memory("After loading decoder")
            
            # Free decoder weights from CPU memory
            del checkpoint['decoder']
            del decoder_state
            torch.cuda.empty_cache()
            
    else:
        print("📦 Using standard PyTorch loading\n")
        
        # Load encoder directly to target device
        print("Creating and loading encoder...")
        encoder = BidirectionalEncoder(encoder_config, device='cpu')
        
        if 'encoder' in checkpoint:
            encoder.load_state_dict(checkpoint['encoder'], strict=True)
            del checkpoint['encoder']
            
        encoder = encoder.to(device=device, dtype=dtype)
        print(f"✓ Encoder: {sum(p.numel() for p in encoder.parameters())/1e6:.2f}M params")
        print_gpu_memory("After loading encoder")
        
        torch.cuda.empty_cache()
        
        # Load decoder directly to target device
        print("\nCreating and loading decoder...")
        decoder = Transformer(decoder_config, device='cpu')
        
        if 'decoder' in checkpoint:
            decoder.load_state_dict(checkpoint['decoder'], strict=True)
            del checkpoint['decoder']
            
        decoder = decoder.to(device=device, dtype=dtype)
        print(f"✓ Decoder: {sum(p.numel() for p in decoder.parameters())/1e6:.2f}M params")
        print_gpu_memory("After loading decoder")
        
        torch.cuda.empty_cache()
    
    # STEP 3: Clean up checkpoint
    print("\n" + "=" * 60)
    print("STEP 3: Cleaning up checkpoint")
    print("=" * 60)
    
    checkpoint.clear()
    del checkpoint
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    print("✓ Checkpoint deleted from memory")
    print_gpu_memory("After cleanup")
    
    # STEP 4: Test inference
    print("\n" + "=" * 60)
    print("STEP 4: Testing inference on MS MARCO")
    print("=" * 60)
    
    encoder.eval()
    decoder.eval()
    
    # Load one example from MS MARCO
    print("Loading MS MARCO dataset...")
    dataset = load_dataset("ms_marco", "v2.1", split="validation", streaming=True)
    
    # Get first example
    example = next(iter(dataset))
    
    # Extract data
    if 'passages' in example and 'query' in example:
        passages = example['passages']
        if isinstance(passages, dict) and 'passage_text' in passages:
            context = passages['passage_text'][0] if isinstance(passages['passage_text'], list) else passages['passage_text']
        else:
            context = "This is a test context."
        
        question = example['query']
    else:
        context = "This is a test context."
        question = "What is this?"
    
    print(f"\nContext: {context[:100]}...")
    print(f"Question: {question}")
    
    # Tokenize
    context_tokens = tokenizer.encode(context)[:256]
    question_tokens = tokenizer.encode(question)[:64]
    
    c_marker = tokenizer.encode("<C>")
    sep_marker = tokenizer.encode("<SEP>")
    a_marker = tokenizer.encode("<A>")
    
    # Build encoder input: <C> Context <SEP> Question
    ctx_tokens = c_marker + context_tokens + sep_marker + question_tokens
    ctx_tensor = torch.tensor(ctx_tokens, dtype=torch.long, device=device)
    
    # Build decoder input: Question <A>
    qa_tokens = question_tokens + a_marker
    qa_tensor = torch.tensor(qa_tokens, dtype=torch.long, device=device)
    
    print(f"\nEncoding context ({len(ctx_tokens)} tokens)...")
    print_gpu_memory("Before encoding")
    
    with torch.no_grad():
        encoder_k, encoder_v = encoder(ctx_tensor, return_encoder_kv=True)
        print(f"✓ Encoder output: K={encoder_k.shape}, V={encoder_v.shape}")
        print_gpu_memory("After encoding")
        
        print("\nDecoding answer...")
        decoder.reset_context()
        logits = decoder(qa_tensor, encoder_k=encoder_k, encoder_v=encoder_v)
        print(f"✓ Decoder output: {logits.shape}")
        print_gpu_memory("After decoding")
        
        # Generate next token
        next_token = logits[-1].argmax().item()
        next_word = tokenizer.decode([next_token])
        
        print(f"\nGenerated next token: {next_word}")
    
    print("\n" + "=" * 60)
    print("✓ INFERENCE TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print_gpu_memory("Final")


if __name__ == "__main__":
    main()