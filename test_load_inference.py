#!/usr/bin/env python3
"""
Simple script to load checkpoint and test inference on MS MARCO.
"""
import argparse
import torch
from datasets import load_dataset

from architecture.config import ModelConfig
from architecture.encoder import BidirectionalEncoder, EncoderConfig
from architecture.transformer import Transformer
from architecture.tokenizer import get_tokenizer


def load_model_from_checkpoint(checkpoint_path, device='cuda:0', dtype=torch.bfloat16):
    """
    Load encoder-decoder model from checkpoint.
    
    Args:
        checkpoint_path: Path to .pt checkpoint file
        device: Device to load to ('cuda:0', 'cpu', etc.)
        dtype: Data type (torch.bfloat16, torch.float32, etc.)
    
    Returns:
        encoder, decoder, tokenizer
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    print(f"Target device: {device}, dtype: {dtype}\n")
    
    # Load checkpoint to CPU first
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    print(f"✓ Checkpoint loaded")
    print(f"  Keys: {list(checkpoint.keys())}\n")
    
    # Get tokenizer first (needed for vocab size)
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.n_vocab
    
    # Get configs
    encoder_config = checkpoint.get('encoder_config')
    if isinstance(encoder_config, dict):
        encoder_config = EncoderConfig(**encoder_config)
    
    # If no encoder config, use medium default
    if encoder_config is None:
        print("⚠️  No encoder_config in checkpoint, using medium default")
        encoder_config = EncoderConfig(
            vocab_size=vocab_size, 
            hidden_size=1024, 
            num_hidden_layers=12,
            num_attention_heads=16, 
            num_key_value_heads=16
        )
    
    decoder_config = checkpoint.get('decoder_config')
    if decoder_config is None:
        # Try alternate key
        decoder_config = checkpoint.get('config')
    
    if isinstance(decoder_config, dict):
        decoder_config = ModelConfig(**decoder_config)
    
    # If no decoder config, use medium default
    if decoder_config is None:
        print("⚠️  No decoder_config in checkpoint, using medium default")
        decoder_config = ModelConfig(
            vocab_size=vocab_size,
            hidden_size=2880,
            num_hidden_layers=12,  # Match checkpoint (12 layers, not 24)
            num_experts=32,
            num_attention_heads=64,
            num_key_value_heads=64,
            use_encoder_decoder_cross_attention=True,
            encoder_hidden_size=1024
        )
    
    print(f"Encoder: {encoder_config.hidden_size}d, {encoder_config.num_hidden_layers} layers")
    print(f"Decoder: {decoder_config.hidden_size}d, {decoder_config.num_hidden_layers} layers\n")
    
    # Create encoder
    print("Creating encoder...")
    encoder = BidirectionalEncoder(encoder_config, device='cpu')
    if 'encoder' in checkpoint:
        encoder.load_state_dict(checkpoint['encoder'])
        del checkpoint['encoder']
    encoder = encoder.to(device=device, dtype=dtype)
    print(f"✓ Encoder loaded: {sum(p.numel() for p in encoder.parameters())/1e6:.2f}M params")
    
    # Create decoder
    print("Creating decoder...")
    decoder = Transformer(decoder_config, device='cpu')
    if 'decoder' in checkpoint:
        decoder.load_state_dict(checkpoint['decoder'])
        del checkpoint['decoder']
    decoder = decoder.to(device=device, dtype=dtype)
    print(f"✓ Decoder loaded: {sum(p.numel() for p in decoder.parameters())/1e6:.2f}M params\n")
    
    # Cleanup
    checkpoint.clear()
    del checkpoint
    torch.cuda.empty_cache()
    
    encoder.eval()
    decoder.eval()
    
    return encoder, decoder, tokenizer


def run_inference(encoder, decoder, tokenizer, context, question, device='cuda:0', max_tokens=50):
    """
    Run inference on a context-question pair.
    
    Args:
        encoder: Encoder model
        decoder: Decoder model
        tokenizer: Tokenizer
        context: Context text
        question: Question text
        device: Device to run on
        max_tokens: Maximum tokens to generate
    
    Returns:
        Generated answer text
    """
    # Tokenize inputs
    context_tokens = tokenizer.encode(context)[:256]
    question_tokens = tokenizer.encode(question)[:64]
    
    c_marker = tokenizer.encode("<C>")
    sep_marker = tokenizer.encode("<SEP>")
    a_marker = tokenizer.encode("<A>")
    
    # Build encoder input: <C> Context <SEP> Question
    ctx_tokens = c_marker + context_tokens + sep_marker + question_tokens
    ctx_tensor = torch.tensor(ctx_tokens, dtype=torch.long, device=device)
    
    # Encode context
    with torch.no_grad():
        encoder_k, encoder_v = encoder(ctx_tensor, return_encoder_kv=True)
    
    # Build initial decoder input: Question <A>
    current_tokens = question_tokens + a_marker
    
    # Generate tokens autoregressively
    generated_tokens = []
    
    with torch.no_grad():
        for _ in range(max_tokens):
            # Create tensor for current sequence
            qa_tensor = torch.tensor(current_tokens, dtype=torch.long, device=device)
            
            # Decode
            decoder.reset_context()
            logits = decoder(qa_tensor, encoder_k=encoder_k, encoder_v=encoder_v)
            
            # Get next token
            next_token = logits[-1].argmax().item()
            
            # Check for end token or padding
            if next_token == 0 or next_token == tokenizer.encode("<|endoftext|>")[0]:
                break
            
            # Add to sequence
            generated_tokens.append(next_token)
            current_tokens.append(next_token)
    
    # Decode generated tokens
    answer = tokenizer.decode(generated_tokens)
    return answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--max_tokens", type=int, default=50, help="Max tokens to generate")
    args = parser.parse_args()
    
    # Convert dtype string to torch dtype
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map[args.dtype]
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Load model
    encoder, decoder, tokenizer = load_model_from_checkpoint(args.checkpoint, device, dtype)
    
    # Load MS MARCO dataset
    print("="*60)
    print("Loading MS MARCO dataset...")
    print("="*60)
    dataset = load_dataset("ms_marco", "v2.1", split="validation", streaming=True)
    
    # Get first example
    example = next(iter(dataset))
    
    # Extract context and question
    if 'passages' in example and 'query' in example:
        passages = example['passages']
        if isinstance(passages, dict) and 'passage_text' in passages:
            context = passages['passage_text'][0] if isinstance(passages['passage_text'], list) else passages['passage_text']
        else:
            context = "This is a test context about machine learning and AI."
        question = example['query']
    else:
        context = "Machine learning is a subset of artificial intelligence."
        question = "What is machine learning?"
    
    print(f"\nContext: {context[:200]}...")
    print(f"Question: {question}\n")
    
    # Run inference
    print("="*60)
    print("Generating answer...")
    print("="*60)
    
    answer = run_inference(encoder, decoder, tokenizer, context, question, device, args.max_tokens)
    
    print(f"\n{'='*60}")
    print(f"RESULT")
    print(f"{'='*60}")
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
