"""
Test script for encoder-decoder cross-attention architecture.

Demonstrates:
1. Encoder processing with chunking and context embedding collection
2. Decoder processing with encoder-decoder cross-attention
3. Context embedding flow between encoder and decoder
"""

import torch
from architecture.encoder import BidirectionalEncoder, EncoderConfig
from architecture.transformer import Transformer
from architecture.config import ModelConfig


def test_encoder_decoder():
    """Test encoder-decoder cross-attention with chunked sequences."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create encoder config
    enc_config = EncoderConfig(
        vocab_size=50257,
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        num_key_value_heads=8,
        head_dim=64,
        intermediate_size=2048,
        max_position_embeddings=128,  # Chunk size for encoder
        use_moe=False
    )
    
    # Create decoder config
    dec_config = ModelConfig(
        vocab_size=50257,
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=64,
        intermediate_size=512,
        sliding_window=128,  # Chunk size for decoder
        use_lsi_cross_attention=False,  # Disable LSI for this test
        use_context_embedding=True,
        use_encoder_decoder_cross_attention=True  # Enable encoder-decoder cross-attention
    )
    
    # Initialize models
    print("\nInitializing encoder...")
    encoder = BidirectionalEncoder(enc_config, device=device)
    print(f"Encoder parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    
    print("\nInitializing decoder...")
    decoder = Transformer(dec_config, device=device)
    print(f"Decoder parameters: {sum(p.numel() for p in decoder.parameters()):,}")
    
    # Test with long sequence (will be chunked)
    print("\n" + "="*80)
    print("TEST 1: Long sequences with chunking")
    print("="*80)
    
    # Encoder input: 256 tokens (will be split into 2 chunks of 128)
    encoder_input = torch.randint(0, 50257, (256,), device=device)
    print(f"\nEncoder input shape: {encoder_input.shape}")
    print(f"Expected chunks: {(256 + 128 - 1) // 128} (chunk_size=128)")
    
    # Encode with SVD compression
    print("\nEncoding with SVD compression...")
    encoder.reset_context()
    encoder_k, encoder_v = encoder(
        encoder_input, 
        return_compressed_kv=True,
        chunk_size=128
    )
    print(f"Encoder K shape: {encoder_k.shape}")
    print(f"Encoder V shape: {encoder_v.shape}")
    print(f"  -> Compressed from {256} tokens to {encoder_k.shape[0]} via SVD")
    
    # Decoder input: 200 tokens (will be padded to 2 chunks of 128)
    decoder_input = torch.randint(0, 50257, (200,), device=device)
    print(f"\nDecoder input shape: {decoder_input.shape}")
    print(f"Expected chunks: {(200 + 128 - 1) // 128} (chunk_size=128, last padded)")
    
    # Decode with encoder K, V
    print("\nDecoding with encoder-decoder cross-attention...")
    decoder.reset_context()
    logits = decoder(
        decoder_input,
        update_context=True,
        max_seq_len=128,
        encoder_k=encoder_k,
        encoder_v=encoder_v
    )
    print(f"Output logits shape: {logits.shape}")
    print(f"  -> Predictions for {logits.shape[0]} tokens, vocab_size={logits.shape[1]}")
    
    # Test with short sequence (no chunking)
    print("\n" + "="*80)
    print("TEST 2: Short sequences without chunking")
    print("="*80)
    
    # Short sequences
    encoder_input_short = torch.randint(0, 50257, (64,), device=device)
    decoder_input_short = torch.randint(0, 50257, (64,), device=device)
    
    print(f"\nEncoder input shape: {encoder_input_short.shape}")
    print(f"Decoder input shape: {decoder_input_short.shape}")
    
    # Encode
    encoder.reset_context()
    encoder_k_short, encoder_v_short = encoder(
        encoder_input_short,
        return_compressed_kv=True,
        chunk_size=128
    )
    print(f"Encoder K shape: {encoder_k_short.shape}")
    print(f"Encoder V shape: {encoder_v_short.shape}")
    
    # Decode
    decoder.reset_context()
    logits_short = decoder(
        decoder_input_short,
        encoder_k=encoder_k_short,
        encoder_v=encoder_v_short
    )
    print(f"Output logits shape: {logits_short.shape}")
    
    # Test attention flow
    print("\n" + "="*80)
    print("TEST 3: Cross-attention information flow")
    print("="*80)
    
    print("\nComparing decoder outputs with and without encoder K, V...")
    
    # Without encoder K, V
    decoder.reset_context()
    logits_without = decoder(decoder_input_short, encoder_k=None, encoder_v=None)
    
    # With encoder K, V
    decoder.reset_context()
    logits_with = decoder(decoder_input_short, encoder_k=encoder_k_short, encoder_v=encoder_v_short)
    
    # Check if outputs differ (they should if cross-attention is working)
    diff = (logits_with - logits_without).abs().mean()
    print(f"Mean absolute difference: {diff.item():.6f}")
    
    if diff.item() > 1e-4:
        print("✓ Cross-attention is affecting decoder outputs (as expected)")
    else:
        print("✗ Cross-attention might not be working (outputs are too similar)")
    
    print("\n" + "="*80)
    print("All tests completed!")
    print("="*80)


if __name__ == "__main__":
    test_encoder_decoder()
