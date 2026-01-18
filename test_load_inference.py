#!/usr/bin/env python3
"""
Simple script to test loading model checkpoint and doing inference.
This helps diagnose GPU memory issues.
"""
import argparse
import torch
from datasets import load_dataset

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Get tokenizer
    tokenizer = get_tokenizer()
    
    # STEP 1: Load checkpoint to CPU
    print("=" * 60)
    print("STEP 1: Loading checkpoint to CPU")
    print("=" * 60)
    print_gpu_memory("Before loading checkpoint")
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    print(f"✓ Checkpoint loaded to CPU")
    print(f"  Keys in checkpoint: {list(checkpoint.keys())}")
    print_gpu_memory("After loading checkpoint to CPU")
    
    # STEP 2: Extract configs
    print("\n" + "=" * 60)
    print("STEP 2: Extracting configs")
    print("=" * 60)
    
    encoder_config = checkpoint.get('encoder_config')
    if isinstance(encoder_config, dict):
        encoder_config = EncoderConfig(**encoder_config)
    
    decoder_config = checkpoint.get('decoder_config')
    if isinstance(decoder_config, dict):
        decoder_config = ModelConfig(**decoder_config)
    
    print(f"Encoder config: {encoder_config.hidden_size}d, {encoder_config.num_hidden_layers} layers")
    print(f"Decoder config: {decoder_config.hidden_size}d, {decoder_config.num_hidden_layers} layers")
    
    # STEP 3: Create encoder on GPU
    print("\n" + "=" * 60)
    print("STEP 3: Creating encoder on GPU")
    print("=" * 60)
    print_gpu_memory("Before creating encoder")
    
    encoder = BidirectionalEncoder(encoder_config, device=device)
    encoder.to(device)
    
    print(f"✓ Encoder created: {sum(p.numel() for p in encoder.parameters())/1e6:.2f}M params")
    print_gpu_memory("After creating encoder")
    
    # STEP 4: Load encoder weights
    print("\n" + "=" * 60)
    print("STEP 4: Loading encoder weights")
    print("=" * 60)
    
    if 'encoder' in checkpoint:
        print("Loading encoder state dict...")
        encoder_state = checkpoint['encoder']
        print(f"  Encoder state dict has {len(encoder_state)} keys")
        print(f"  First few keys: {list(encoder_state.keys())[:3]}")
        
        # Load parameter by parameter
        for name, param in encoder.named_parameters():
            if name in encoder_state:
                param.data.copy_(encoder_state[name].to(device))
        
        for name, buffer in encoder.named_buffers():
            if name in encoder_state:
                buffer.data.copy_(encoder_state[name].to(device))
        
        print("✓ Encoder weights loaded")
        del checkpoint['encoder']
        print_gpu_memory("After loading encoder weights")
    
    torch.cuda.empty_cache()
    print_gpu_memory("After clearing cache")
    
    # STEP 5: Create decoder on GPU
    print("\n" + "=" * 60)
    print("STEP 5: Creating decoder on GPU")
    print("=" * 60)
    print_gpu_memory("Before creating decoder")
    
    decoder = Transformer(decoder_config, device=device)
    decoder.to(device)
    
    print(f"✓ Decoder created: {sum(p.numel() for p in decoder.parameters())/1e6:.2f}M params")
    print_gpu_memory("After creating decoder")
    
    # STEP 6: Load decoder weights
    print("\n" + "=" * 60)
    print("STEP 6: Loading decoder weights")
    print("=" * 60)
    
    if 'decoder' in checkpoint:
        print("Loading decoder state dict...")
        decoder_state = checkpoint['decoder']
        print(f"  Decoder state dict has {len(decoder_state)} keys")
        
        # Load parameter by parameter
        for name, param in decoder.named_parameters():
            if name in decoder_state:
                param.data.copy_(decoder_state[name].to(device))
        
        for name, buffer in decoder.named_buffers():
            if name in decoder_state:
                buffer.data.copy_(decoder_state[name].to(device))
        
        print("✓ Decoder weights loaded")
        del checkpoint['decoder']
        print_gpu_memory("After loading decoder weights")
    
    torch.cuda.empty_cache()
    print_gpu_memory("After clearing cache")
    
    # STEP 7: Delete checkpoint completely
    print("\n" + "=" * 60)
    print("STEP 7: Cleaning up checkpoint")
    print("=" * 60)
    
    checkpoint.clear()
    del checkpoint
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    print("✓ Checkpoint deleted from memory")
    print_gpu_memory("After deleting checkpoint")
    
    # STEP 8: Load test data and do inference
    print("\n" + "=" * 60)
    print("STEP 8: Testing inference on MS MARCO")
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
