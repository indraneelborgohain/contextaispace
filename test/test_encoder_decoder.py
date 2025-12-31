"""
Test cases for encoder-decoder architecture with cross-attention

Tests various scenarios:
- Different context lengths (128, 512)
- Variable question and answer lengths
- SVD compression
- Cross-attention integration
"""

import pytest
import torch
from architecture.config import ModelConfig
from architecture.encoder import BidirectionalEncoder
from architecture.transformer import Transformer


class TestEncoderDecoder:
    """Test encoder-decoder integration with various input sizes"""
    
    @pytest.fixture
    def device(self):
        """Get device for testing"""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @pytest.fixture
    def config(self):
        """Create a small test configuration"""
        return ModelConfig(
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=4,
            head_dim=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_experts=4,
            experts_per_token=2,
            intermediate_size=256,
            sliding_window=128,
            initial_context_length=512,
            use_encoder_decoder_cross_attention=True,
            use_lsi_cross_attention=False,
            use_context_embedding=True,
        )
    
    @pytest.fixture
    def encoder(self, config, device):
        """Create encoder instance"""
        return BidirectionalEncoder(config, device=device)
    
    @pytest.fixture
    def decoder(self, config, device):
        """Create decoder instance"""
        return Transformer(config, device=device)
    
    def test_encoder_short_context(self, encoder, config, device):
        """Test encoder with short context (< window size)"""
        context_len = 64
        context_tokens = torch.randint(0, config.vocab_size, (context_len,), device=device)
        
        encoder_k, encoder_v = encoder(context_tokens, return_compressed_kv=True)
        
        # Check output shapes
        assert encoder_k.shape == (config.sliding_window, config.hidden_size)
        assert encoder_v.shape == (config.sliding_window, config.hidden_size)
        print(f"✓ Short context ({context_len} tokens) → compressed to {encoder_k.shape}")
    
    def test_encoder_medium_context(self, encoder, config, device):
        """Test encoder with medium context (= window size)"""
        context_len = 128
        context_tokens = torch.randint(0, config.vocab_size, (context_len,), device=device)
        
        encoder_k, encoder_v = encoder(context_tokens, return_compressed_kv=True)
        
        # Check output shapes
        assert encoder_k.shape == (config.sliding_window, config.hidden_size)
        assert encoder_v.shape == (config.sliding_window, config.hidden_size)
        print(f"✓ Medium context ({context_len} tokens) → compressed to {encoder_k.shape}")
    
    def test_encoder_long_context(self, encoder, config, device):
        """Test encoder with long context (> window size)"""
        context_len = 512
        context_tokens = torch.randint(0, config.vocab_size, (context_len,), device=device)
        
        encoder_k, encoder_v = encoder(context_tokens, return_compressed_kv=True)
        
        # Check output shapes - should compress multiple chunks to window size
        assert encoder_k.shape == (config.sliding_window, config.hidden_size)
        assert encoder_v.shape == (config.sliding_window, config.hidden_size)
        print(f"✓ Long context ({context_len} tokens) → compressed to {encoder_k.shape}")
    
    def test_decoder_with_encoder_short_qa(self, decoder, encoder, config, device):
        """Test decoder with encoder on short Q&A"""
        # Encode context
        context_len = 128
        context_tokens = torch.randint(0, config.vocab_size, (context_len,), device=device)
        encoder_k, encoder_v = encoder(context_tokens, return_compressed_kv=True)
        
        # Decode question + answer
        qa_len = 32
        qa_tokens = torch.randint(0, config.vocab_size, (qa_len,), device=device)
        
        decoder.reset_context()
        logits = decoder(qa_tokens, encoder_k=encoder_k, encoder_v=encoder_v)
        
        # Check output shape
        assert logits.shape == (qa_len, config.vocab_size)
        print(f"✓ Context {context_len} + Q&A {qa_len} → logits {logits.shape}")
    
    def test_decoder_with_encoder_medium_qa(self, decoder, encoder, config, device):
        """Test decoder with encoder on medium Q&A"""
        # Encode context
        context_len = 256
        context_tokens = torch.randint(0, config.vocab_size, (context_len,), device=device)
        encoder_k, encoder_v = encoder(context_tokens, return_compressed_kv=True)
        
        # Decode question + answer
        qa_len = 100
        qa_tokens = torch.randint(0, config.vocab_size, (qa_len,), device=device)
        
        decoder.reset_context()
        logits = decoder(qa_tokens, encoder_k=encoder_k, encoder_v=encoder_v)
        
        # Check output shape
        assert logits.shape == (qa_len, config.vocab_size)
        print(f"✓ Context {context_len} + Q&A {qa_len} → logits {logits.shape}")
    
    def test_decoder_with_encoder_long_qa(self, decoder, encoder, config, device):
        """Test decoder with encoder on long Q&A (multiple chunks)"""
        # Encode context
        context_len = 512
        context_tokens = torch.randint(0, config.vocab_size, (context_len,), device=device)
        encoder_k, encoder_v = encoder(context_tokens, return_compressed_kv=True)
        
        # Decode long question + answer (will chunk)
        qa_len = 200
        qa_tokens = torch.randint(0, config.vocab_size, (qa_len,), device=device)
        
        decoder.reset_context()
        logits = decoder(qa_tokens, encoder_k=encoder_k, encoder_v=encoder_v)
        
        # Check output shape
        assert logits.shape == (qa_len, config.vocab_size)
        print(f"✓ Context {context_len} + Q&A {qa_len} (chunked) → logits {logits.shape}")
    
    def test_decoder_without_encoder(self, decoder, config, device):
        """Test decoder works without encoder (standard mode)"""
        qa_len = 64
        qa_tokens = torch.randint(0, config.vocab_size, (qa_len,), device=device)
        
        decoder.reset_context()
        logits = decoder(qa_tokens)
        
        # Check output shape
        assert logits.shape == (qa_len, config.vocab_size)
        print(f"✓ Decoder-only mode: Q&A {qa_len} → logits {logits.shape}")
    
    def test_cross_attention_effect(self, decoder, encoder, config, device):
        """Test that cross-attention actually affects output"""
        context_len = 128
        qa_len = 50
        
        context_tokens = torch.randint(0, config.vocab_size, (context_len,), device=device)
        qa_tokens = torch.randint(0, config.vocab_size, (qa_len,), device=device)
        
        # Get encoder outputs
        encoder_k, encoder_v = encoder(context_tokens, return_compressed_kv=True)
        
        # Forward with cross-attention
        decoder.reset_context()
        logits_with_encoder = decoder(qa_tokens, encoder_k=encoder_k, encoder_v=encoder_v)
        
        # Forward without cross-attention
        decoder.reset_context()
        logits_without_encoder = decoder(qa_tokens)
        
        # Outputs should be different
        diff = torch.abs(logits_with_encoder - logits_without_encoder).mean().item()
        assert diff > 1e-6, "Cross-attention should affect output"
        print(f"✓ Cross-attention effect: mean diff = {diff:.6f}")
    
    def test_svd_compression_quality(self, encoder, config, device):
        """Test that SVD compression preserves information"""
        context_len = 256  # 2 chunks
        context_tokens = torch.randint(0, config.vocab_size, (context_len,), device=device)
        
        # Get compressed outputs
        encoder_k, encoder_v = encoder(context_tokens, return_compressed_kv=True)
        
        # Check that we don't have NaN or Inf
        assert not torch.isnan(encoder_k).any(), "Encoder K contains NaN"
        assert not torch.isnan(encoder_v).any(), "Encoder V contains NaN"
        assert not torch.isinf(encoder_k).any(), "Encoder K contains Inf"
        assert not torch.isinf(encoder_v).any(), "Encoder V contains Inf"
        
        # Check reasonable value ranges
        k_mean = encoder_k.abs().mean().item()
        v_mean = encoder_v.abs().mean().item()
        assert 0 < k_mean < 100, f"Encoder K mean {k_mean} out of range"
        assert 0 < v_mean < 100, f"Encoder V mean {v_mean} out of range"
        
        print(f"✓ SVD compression quality: K_mean={k_mean:.4f}, V_mean={v_mean:.4f}")
    
    def test_variable_lengths(self, decoder, encoder, config, device):
        """Test with various combinations of context and Q&A lengths"""
        test_cases = [
            (64, 20),    # short context, short Q&A
            (128, 50),   # medium context, medium Q&A
            (256, 100),  # long context, long Q&A
            (512, 150),  # very long context, very long Q&A
            (100, 200),  # short context, long Q&A
            (300, 30),   # long context, short Q&A
        ]
        
        for context_len, qa_len in test_cases:
            context_tokens = torch.randint(0, config.vocab_size, (context_len,), device=device)
            qa_tokens = torch.randint(0, config.vocab_size, (qa_len,), device=device)
            
            encoder_k, encoder_v = encoder(context_tokens, return_compressed_kv=True)
            
            decoder.reset_context()
            logits = decoder(qa_tokens, encoder_k=encoder_k, encoder_v=encoder_v)
            
            assert logits.shape == (qa_len, config.vocab_size)
            print(f"✓ Context {context_len:3d} + Q&A {qa_len:3d} → OK")
    
    def test_gradient_flow(self, decoder, encoder, config, device):
        """Test that gradients flow through encoder-decoder"""
        context_len = 128
        qa_len = 50
        
        context_tokens = torch.randint(0, config.vocab_size, (context_len,), device=device)
        qa_tokens = torch.randint(0, config.vocab_size, (qa_len,), device=device)
        targets = torch.randint(0, config.vocab_size, (qa_len,), device=device)
        
        # Forward pass
        encoder_k, encoder_v = encoder(context_tokens, return_compressed_kv=True)
        
        decoder.reset_context()
        logits = decoder(qa_tokens, encoder_k=encoder_k, encoder_v=encoder_v)
        
        # Compute loss
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, config.vocab_size),
            targets.view(-1)
        )
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        encoder_has_grads = any(p.grad is not None for p in encoder.parameters())
        decoder_has_grads = any(p.grad is not None for p in decoder.parameters())
        
        assert encoder_has_grads, "Encoder should have gradients"
        assert decoder_has_grads, "Decoder should have gradients"
        print(f"✓ Gradients flow through encoder-decoder: loss={loss.item():.4f}")


def run_tests():
    """Run all tests with detailed output"""
    print("=" * 80)
    print("ENCODER-DECODER INTEGRATION TESTS")
    print("=" * 80)
    
    # Create test instance
    test = TestEncoderDecoder()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}\n")
    
    config = test.config()
    encoder = test.encoder(config, device)
    decoder = test.decoder(config, device)
    
    print("\n--- Encoder Tests ---")
    test.test_encoder_short_context(encoder, config, device)
    test.test_encoder_medium_context(encoder, config, device)
    test.test_encoder_long_context(encoder, config, device)
    
    print("\n--- Decoder with Encoder Tests ---")
    test.test_decoder_with_encoder_short_qa(decoder, encoder, config, device)
    test.test_decoder_with_encoder_medium_qa(decoder, encoder, config, device)
    test.test_decoder_with_encoder_long_qa(decoder, encoder, config, device)
    
    print("\n--- Decoder Standalone Test ---")
    test.test_decoder_without_encoder(decoder, config, device)
    
    print("\n--- Cross-Attention Effect Test ---")
    test.test_cross_attention_effect(decoder, encoder, config, device)
    
    print("\n--- SVD Compression Test ---")
    test.test_svd_compression_quality(encoder, config, device)
    
    print("\n--- Variable Length Tests ---")
    test.test_variable_lengths(decoder, encoder, config, device)
    
    print("\n--- Gradient Flow Test ---")
    test.test_gradient_flow(decoder, encoder, config, device)
    
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED ✓")
    print("=" * 80)


if __name__ == "__main__":
    run_tests()
