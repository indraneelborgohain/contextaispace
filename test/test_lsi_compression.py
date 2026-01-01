#!/usr/bin/env python3
"""
Test LSI compression in encoder
"""
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from architecture.encoder import BidirectionalEncoder, EncoderConfig
from architecture.tokenizer import get_tokenizer


def test_lsi_compression():
    """Test LSI compression with variable-length sequences"""
    print("=" * 80)
    print("Testing LSI Cross-Attention Compression")
    print("=" * 80)
    
    device = torch.device("cpu")
    tokenizer = get_tokenizer()
    
    # Create encoder with LSI compression
    config = EncoderConfig(
        vocab_size=tokenizer.n_vocab,
        hidden_size=256,
        num_hidden_layers=2,
        head_dim=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=2,
        experts_per_token=1,
        intermediate_size=256,
        max_position_embeddings=512,
        use_lsi_compression=True,
        num_compression_slots=32,  # Fixed compression size
    )
    
    print(f"\nEncoder Config:")
    print(f"  - Hidden size: {config.hidden_size}")
    print(f"  - Num layers: {config.num_hidden_layers}")
    print(f"  - LSI compression: {config.use_lsi_compression}")
    print(f"  - Compression slots: {config.num_compression_slots}")
    
    encoder = BidirectionalEncoder(config, device=device)
    encoder.eval()
    
    # Test with different sequence lengths
    test_cases = [
        ("Short sequence", 50),
        ("Medium sequence", 200),
        ("Long sequence", 500),
        ("Very long sequence", 1000),
    ]
    
    print("\n" + "=" * 80)
    print("Testing Variable-Length Sequences")
    print("=" * 80)
    
    for name, seq_len in test_cases:
        print(f"\n{name} ({seq_len} tokens):")
        
        # Create random token sequence
        tokens = torch.randint(0, 1000, (seq_len,), device=device)
        
        # Encode and compress
        with torch.no_grad():
            k_compressed, v_compressed = encoder(
                tokens,
                return_compressed_kv=True,
                chunk_size=128
            )
        
        print(f"  Input shape: {tokens.shape}")
        print(f"  Output K shape: {k_compressed.shape}")
        print(f"  Output V shape: {v_compressed.shape}")
        print(f"  ✓ Compressed to fixed size: {config.num_compression_slots} slots")
        
        # Verify shape
        assert k_compressed.shape == (config.num_compression_slots, config.hidden_size), \
            f"Expected ({config.num_compression_slots}, {config.hidden_size}), got {k_compressed.shape}"
        assert v_compressed.shape == (config.num_compression_slots, config.hidden_size), \
            f"Expected ({config.num_compression_slots}, {config.hidden_size}), got {v_compressed.shape}"
    
    print("\n" + "=" * 80)
    print("✓ All LSI compression tests passed!")
    print("=" * 80)


def test_svd_vs_lsi():
    """Compare SVD and LSI compression"""
    print("\n" + "=" * 80)
    print("Comparing SVD vs LSI Compression")
    print("=" * 80)
    
    device = torch.device("cpu")
    tokenizer = get_tokenizer()
    
    # Test sequence
    seq_len = 512
    tokens = torch.randint(0, 1000, (seq_len,), device=device)
    
    # SVD encoder
    config_svd = EncoderConfig(
        vocab_size=tokenizer.n_vocab,
        hidden_size=256,
        num_hidden_layers=2,
        head_dim=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        use_lsi_compression=False,  # SVD
        num_compression_slots=64,
    )
    encoder_svd = BidirectionalEncoder(config_svd, device=device)
    encoder_svd.eval()
    
    # LSI encoder
    config_lsi = EncoderConfig(
        vocab_size=tokenizer.n_vocab,
        hidden_size=256,
        num_hidden_layers=2,
        head_dim=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        use_lsi_compression=True,  # LSI
        num_compression_slots=64,
    )
    encoder_lsi = BidirectionalEncoder(config_lsi, device=device)
    encoder_lsi.eval()
    
    print(f"\nInput: {seq_len} tokens")
    print(f"Target compression: 64 slots")
    
    with torch.no_grad():
        k_svd, v_svd = encoder_svd(tokens, return_compressed_kv=True, chunk_size=128)
        k_lsi, v_lsi = encoder_lsi(tokens, return_compressed_kv=True, chunk_size=128)
    
    print(f"\nSVD Compression:")
    print(f"  - Output shape: {k_svd.shape}")
    print(f"  - Method: Singular Value Decomposition (deterministic)")
    print(f"  - Parameters: 0 (no learning)")
    
    print(f"\nLSI Compression:")
    print(f"  - Output shape: {k_lsi.shape}")
    print(f"  - Method: Learned cross-attention with latent slots")
    lsi_params = sum(p.numel() for p in encoder_lsi.lsi_compression.parameters())
    print(f"  - Parameters: {lsi_params:,} (trainable)")
    
    # Count compression module parameters
    latent_slots = config_lsi.num_compression_slots * config_lsi.hidden_size
    print(f"    * Latent slots: {latent_slots:,}")
    print(f"    * Projection layers: {lsi_params - latent_slots:,}")
    
    assert k_svd.shape == k_lsi.shape, "SVD and LSI should produce same output shape"
    
    print("\n✓ Both methods produce fixed-size output: 64 slots")
    print("=" * 80)


if __name__ == "__main__":
    test_lsi_compression()
    test_svd_vs_lsi()
    print("\n🎉 All tests passed!")
