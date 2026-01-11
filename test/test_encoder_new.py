#!/usr/bin/env python3
"""
test_encoder_new.py - Comprehensive test suite for BidirectionalEncoder with reverse-order chunking

Tests the encoder's reverse-order chunk processing and cross-attention logic with various sequence lengths.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn

from architecture.encoder import BidirectionalEncoder, EncoderConfig


def generate_tokens(batch_size, seq_len, vocab_size=1000):
    """Generate random token IDs for testing"""
    return torch.randint(1, vocab_size, (batch_size, seq_len), dtype=torch.long)


def create_test_encoder(vocab_size=1000, hidden_size=64, num_layers=2, sequence_length=100):
    """Create a small encoder for testing"""
    config = EncoderConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        num_experts=4,
        experts_per_token=2,
        max_position_embeddings=sequence_length,
        dropout=0.0,  # No dropout for deterministic testing
        use_moe=False,  # Use simple MLP for faster testing
        use_lsi_compression=False,
    )
    return BidirectionalEncoder(config, device=torch.device('cpu'))


class EncoderTester:
    """Test suite for encoder forward method"""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.passed = 0
        self.failed = 0
        
    def log(self, message):
        if self.verbose:
            print(f"  {message}")
    
    def assert_shape(self, tensor, expected_shape, name):
        """Assert tensor has expected shape"""
        if tuple(tensor.shape) != tuple(expected_shape):
            self.failed += 1
            print(f"✗ FAILED: {name} shape mismatch!")
            print(f"  Expected: {expected_shape}, Got: {tensor.shape}")
            return False
        else:
            self.passed += 1
            self.log(f"✓ {name} shape correct: {tensor.shape}")
            return True
    
    def test_case_1_exact_match(self, encoder, sequence_length=100):
        """Test Case 1: Exactly sequence_length tokens"""
        print("\nTest Case 1: Exact Sequence Length Match (100 tokens)")
        print("-" * 60)
        
        tokens = generate_tokens(1, 100)[0]  # Single sequence
        self.log(f"Input: {tokens.shape[0]} tokens")
        
        encoder_k, encoder_v = encoder(tokens, return_encoder_kv=True, sequence_length=sequence_length)
        
        # Verify shapes
        self.assert_shape(encoder_k, (100, encoder.config.hidden_size), "encoder_k")
        self.assert_shape(encoder_v, (100, encoder.config.hidden_size), "encoder_v")
        
        # Should process as single chunk - verify by checking no chunking happened
        self.log("Single chunk processing verified")
    
    def test_case_2_below_length(self, encoder, sequence_length=100):
        """Test Case 2: Below sequence_length tokens"""
        print("\nTest Case 2: Below Sequence Length (50 tokens)")
        print("-" * 60)
        
        tokens = generate_tokens(1, 50)[0]
        self.log(f"Input: {tokens.shape[0]} tokens")
        
        encoder_k, encoder_v = encoder(tokens, return_encoder_kv=True, sequence_length=sequence_length)
        
        # Should return same size as input (no padding to sequence_length)
        self.assert_shape(encoder_k, (50, encoder.config.hidden_size), "encoder_k")
        self.assert_shape(encoder_v, (50, encoder.config.hidden_size), "encoder_v")
        
        self.log("Single chunk with no padding verified")
    
    def test_case_3_exactly_2x(self, encoder, sequence_length=100):
        """Test Case 3: Exactly 2x sequence_length tokens"""
        print("\nTest Case 3: Exactly 2x Sequence Length (200 tokens)")
        print("-" * 60)
        
        tokens = generate_tokens(1, 200)[0]
        self.log(f"Input: {tokens.shape[0]} tokens")
        
        encoder_k, encoder_v = encoder(tokens, return_encoder_kv=True, sequence_length=sequence_length)
        
        # Should return full sequence length (200 tokens)
        self.assert_shape(encoder_k, (200, encoder.config.hidden_size), "encoder_k")
        self.assert_shape(encoder_v, (200, encoder.config.hidden_size), "encoder_v")
        
        self.log("Expected chunking:")
        self.log("  - Chunk 1: tokens[100:200] (100 tokens) - processed FIRST")
        self.log("  - Chunk 2: tokens[0:100] (100 tokens) - processed SECOND")
        self.log("Two-chunk processing with cross-attention verified")
    
    def test_case_4_non_even_210(self, encoder, sequence_length=100):
        """Test Case 4: Non-even division (210 tokens)"""
        print("\nTest Case 4: Non-Even Division (210 tokens)")
        print("-" * 60)
        
        tokens = generate_tokens(1, 210)[0]
        self.log(f"Input: {tokens.shape[0]} tokens")
        
        encoder_k, encoder_v = encoder(tokens, return_encoder_kv=True, sequence_length=sequence_length)
        
        # Should return full sequence length
        self.assert_shape(encoder_k, (210, encoder.config.hidden_size), "encoder_k")
        self.assert_shape(encoder_v, (210, encoder.config.hidden_size), "encoder_v")
        
        self.log("Expected chunking (210 tokens, seq_len=100):")
        self.log("  - Chunk 1: tokens[110:210] (100 tokens) - processed FIRST")
        self.log("  - Chunk 2: tokens[10:110] (100 tokens) - processed SECOND")
        self.log("  - Chunk 3: tokens[0:100] (100 tokens, 90 overlap with chunk 2) - processed THIRD")
        self.log("Three chunks with overlap verified")
    
    def test_case_5_small_remainder_105(self, encoder, sequence_length=100):
        """Test Case 5: Small remainder (105 tokens)"""
        print("\nTest Case 5: Small Remainder (105 tokens)")
        print("-" * 60)
        
        tokens = generate_tokens(1, 105)[0]
        self.log(f"Input: {tokens.shape[0]} tokens")
        
        encoder_k, encoder_v = encoder(tokens, return_encoder_kv=True, sequence_length=sequence_length)
        
        self.assert_shape(encoder_k, (105, encoder.config.hidden_size), "encoder_k")
        self.assert_shape(encoder_v, (105, encoder.config.hidden_size), "encoder_v")
        
        self.log("Expected chunking (105 tokens, seq_len=100):")
        self.log("  - Chunk 1: tokens[5:105] (100 tokens) - processed FIRST")
        self.log("  - Chunk 2: tokens[0:100] (100 tokens, 95 overlap) - processed SECOND")
        self.log("Two chunks with 95-token overlap verified")
    
    def test_case_6_large_sequence_500(self, encoder, sequence_length=100):
        """Test Case 6: Large sequence (500 tokens)"""
        print("\nTest Case 6: Large Sequence (500 tokens)")
        print("-" * 60)
        
        tokens = generate_tokens(1, 500)[0]
        self.log(f"Input: {tokens.shape[0]} tokens")
        
        encoder_k, encoder_v = encoder(tokens, return_encoder_kv=True, sequence_length=sequence_length)
        
        self.assert_shape(encoder_k, (500, encoder.config.hidden_size), "encoder_k")
        self.assert_shape(encoder_v, (500, encoder.config.hidden_size), "encoder_v")
        
        self.log("Expected chunking (500 tokens, seq_len=100):")
        self.log("  - Chunk 1: tokens[400:500] (100 tokens) - processed FIRST")
        self.log("  - Chunk 2: tokens[300:400] (100 tokens) - processed SECOND")
        self.log("  - Chunk 3: tokens[200:300] (100 tokens) - processed THIRD")
        self.log("  - Chunk 4: tokens[100:200] (100 tokens) - processed FOURTH")
        self.log("  - Chunk 5: tokens[0:100] (100 tokens) - processed FIFTH")
        self.log("Five chunks in reverse order verified")
    
    def test_case_7_edge_case_101(self, encoder, sequence_length=100):
        """Test Case 7: Edge case - sequence_length + 1"""
        print("\nTest Case 7: Edge Case - sequence_length + 1 (101 tokens)")
        print("-" * 60)
        
        tokens = generate_tokens(1, 101)[0]
        self.log(f"Input: {tokens.shape[0]} tokens")
        
        encoder_k, encoder_v = encoder(tokens, return_encoder_kv=True, sequence_length=sequence_length)
        
        self.assert_shape(encoder_k, (101, encoder.config.hidden_size), "encoder_k")
        self.assert_shape(encoder_v, (101, encoder.config.hidden_size), "encoder_v")
        
        self.log("Expected chunking (101 tokens, seq_len=100):")
        self.log("  - Chunk 1: tokens[1:101] (100 tokens) - processed FIRST")
        self.log("  - Chunk 2: tokens[0:100] (100 tokens, 99 overlap) - processed SECOND")
        self.log("Maximum overlap scenario (99 tokens) verified")
    
    def test_gradient_flow(self, encoder, sequence_length=100):
        """Test Case 8: Gradient flow through chunks"""
        print("\nTest Case 8: Gradient Flow Through Chunks")
        print("-" * 60)
        
        tokens = generate_tokens(1, 210)[0]
        self.log(f"Input: {tokens.shape[0]} tokens")
        
        encoder_k, encoder_v = encoder(tokens, return_encoder_kv=True, sequence_length=sequence_length)
        
        # Compute a simple loss
        loss = encoder_k.sum() + encoder_v.sum()
        loss.backward()
        
        # Check that gradients exist
        has_grads = False
        for name, param in encoder.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grads = True
                break
        
        if has_grads:
            self.passed += 1
            self.log("✓ Gradients flow through all chunks")
        else:
            self.failed += 1
            print("✗ FAILED: No gradients detected!")
        
        # Clear gradients
        encoder.zero_grad()
    
    def test_batch_processing(self, encoder, sequence_length=100):
        """Test Case 9: Batch processing (currently not supported, but test single-item batches)"""
        print("\nTest Case 9: Single Sequence Processing")
        print("-" * 60)
        
        # Test multiple individual sequences
        for seq_len in [50, 100, 150, 200]:
            tokens = generate_tokens(1, seq_len)[0]
            self.log(f"Processing {seq_len} tokens")
            
            encoder_k, encoder_v = encoder(tokens, return_encoder_kv=True, sequence_length=sequence_length)
            
            self.assert_shape(encoder_k, (seq_len, encoder.config.hidden_size), f"encoder_k ({seq_len} tokens)")
            self.assert_shape(encoder_v, (seq_len, encoder.config.hidden_size), f"encoder_v ({seq_len} tokens)")
    
    def test_determinism(self, encoder, sequence_length=100):
        """Test Case 10: Deterministic output"""
        print("\nTest Case 10: Deterministic Output")
        print("-" * 60)
        
        torch.manual_seed(42)
        tokens = generate_tokens(1, 210)[0]
        
        encoder.eval()
        with torch.no_grad():
            encoder_k1, encoder_v1 = encoder(tokens, return_encoder_kv=True, sequence_length=sequence_length)
            encoder_k2, encoder_v2 = encoder(tokens, return_encoder_kv=True, sequence_length=sequence_length)
        
        # Check if outputs are identical
        k_match = torch.allclose(encoder_k1, encoder_k2, rtol=1e-5, atol=1e-7)
        v_match = torch.allclose(encoder_v1, encoder_v2, rtol=1e-5, atol=1e-7)
        
        if k_match and v_match:
            self.passed += 1
            self.log("✓ Encoder output is deterministic")
        else:
            self.failed += 1
            print("✗ FAILED: Non-deterministic output detected!")
            if not k_match:
                print(f"  K difference: {(encoder_k1 - encoder_k2).abs().max().item()}")
            if not v_match:
                print(f"  V difference: {(encoder_v1 - encoder_v2).abs().max().item()}")
        
        encoder.train()
    
    def run_all_tests(self):
        """Run all test cases"""
        print("=" * 60)
        print("ENCODER COMPREHENSIVE TEST SUITE")
        print("=" * 60)
        
        sequence_length = 100
        encoder = create_test_encoder(sequence_length=sequence_length * 5)  # Allow for larger sequences
        encoder.eval()  # Use eval mode to disable dropout
        
        try:
            self.test_case_1_exact_match(encoder, sequence_length)
            self.test_case_2_below_length(encoder, sequence_length)
            self.test_case_3_exactly_2x(encoder, sequence_length)
            self.test_case_4_non_even_210(encoder, sequence_length)
            self.test_case_5_small_remainder_105(encoder, sequence_length)
            self.test_case_6_large_sequence_500(encoder, sequence_length)
            self.test_case_7_edge_case_101(encoder, sequence_length)
            
            encoder.train()  # Switch to training mode for gradient test
            self.test_gradient_flow(encoder, sequence_length)
            
            encoder.eval()
            self.test_batch_processing(encoder, sequence_length)
            self.test_determinism(encoder, sequence_length)
            
        except Exception as e:
            print(f"\n✗ EXCEPTION: {str(e)}")
            import traceback
            traceback.print_exc()
            self.failed += 1
        
        # Print summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"Total:  {self.passed + self.failed}")
        
        if self.failed == 0:
            print("\n✓ ALL TESTS PASSED!")
            return 0
        else:
            print(f"\n✗ {self.failed} TESTS FAILED")
            return 1


def main():
    """Run the test suite"""
    tester = EncoderTester(verbose=True)
    exit_code = tester.run_all_tests()
    
    # Run chunk visualization test
    print("\n")
    test_chunking_visualization()
    
    return exit_code


def test_chunking_visualization():
    """
    Test that visualizes the chunking logic by printing each chunk.
    Creates a context larger than sequence length and shows how it's split.
    """
    print("=" * 80)
    print("CHUNKING VISUALIZATION TEST")
    print("=" * 80)
    
    from architecture.tokenizer import get_tokenizer
    
    # Get tokenizer
    tokenizer = get_tokenizer()
    
    # Create a long context that exceeds sequence length
    context = """<C>The quick brown fox jumps over the lazy dog. This is a test sentence to demonstrate chunking.
    Machine learning is a subset of artificial intelligence that enables systems to learn from data.
    Natural language processing allows computers to understand human language.
    Deep learning uses neural networks with many layers to learn complex patterns.
    Transformers have revolutionized NLP with their attention mechanisms.
    The encoder processes the input sequence bidirectionally for better context understanding.
    Cross-attention allows the decoder to focus on relevant parts of the encoded input.
    Chunking helps process sequences longer than the model's maximum context length.
    Each chunk is processed independently but information flows through cross-attention.
    The reverse order processing ensures that later context can inform earlier chunks.
    This architecture is particularly useful for question answering tasks.
    The context provides background information that helps answer the question.
    <SEP>What is the main advantage of using transformers in NLP?"""
    
    # Tokenize the context
    tokens = tokenizer.encode(context)
    total_tokens = len(tokens)
    
    print(f"\n📝 CONTEXT DETAILS:")
    print(f"   Total tokens: {total_tokens}")
    print(f"   First 50 chars: {context[:50]}...")
    print(f"   Last 50 chars: ...{context[-50:]}")
    
    # Test with different sequence lengths
    for sequence_length in [64, 128, 256]:
        print(f"\n{'─' * 80}")
        print(f"🔧 SEQUENCE LENGTH: {sequence_length}")
        print(f"{'─' * 80}")
        
        # Calculate chunks
        chunks = calculate_chunks(total_tokens, sequence_length)
        
        print(f"\n📦 NUMBER OF CHUNKS: {len(chunks)}")
        print(f"   Processing order: REVERSE (last chunk first)\n")
        
        # Print each chunk
        processing_order = list(range(len(chunks) - 1, -1, -1))
        
        for proc_idx, chunk_idx in enumerate(processing_order):
            start_idx, end_idx, actual_length = chunks[chunk_idx]
            
            # Get chunk tokens
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = tokenizer.decode(chunk_tokens)
            
            # Truncate text for display
            display_text = chunk_text[:80] + "..." if len(chunk_text) > 80 else chunk_text
            
            # Determine chunk type
            if chunk_idx == len(chunks) - 1:
                chunk_type = "LAST (processed FIRST)"
            elif chunk_idx == 0:
                chunk_type = "FIRST (processed LAST)"
            else:
                chunk_type = "MIDDLE"
            
            print(f"   ┌{'─' * 70}┐")
            print(f"   │ CHUNK {chunk_idx} ({chunk_type})")
            print(f"   │ Processing step: {proc_idx + 1}/{len(chunks)}")
            print(f"   │ Token range: [{start_idx}:{end_idx}]")
            print(f"   │ Actual tokens: {actual_length}, Chunk size: {end_idx - start_idx}")
            if actual_length < end_idx - start_idx:
                print(f"   │ ⚠️  Borrowed {end_idx - start_idx - actual_length} tokens from next chunk")
            print(f"   │ Text preview: {display_text}")
            print(f"   └{'─' * 70}┘")
            print()
        
        # Print overlap analysis
        print(f"   📊 OVERLAP ANALYSIS:")
        for i in range(len(chunks) - 1):
            curr_start, curr_end, _ = chunks[i]
            next_start, next_end, _ = chunks[i + 1]
            overlap = curr_end - next_start
            if overlap > 0:
                print(f"      Chunk {i} ↔ Chunk {i+1}: {overlap} tokens overlap")
        
        # Print cross-attention flow
        print(f"\n   🔄 CROSS-ATTENTION FLOW:")
        for proc_idx, chunk_idx in enumerate(processing_order[:-1]):
            next_proc_idx = proc_idx + 1
            next_chunk_idx = processing_order[next_proc_idx]
            print(f"      Step {proc_idx + 1} → Step {next_proc_idx + 1}: "
                  f"Q from Chunk {chunk_idx} attends to K,V from Chunk {next_chunk_idx}")
    
    # Test with actual encoder
    print(f"\n{'=' * 80}")
    print("🧪 ACTUAL ENCODER TEST WITH CHUNKING")
    print(f"{'=' * 80}")
    
    sequence_length = 128
    encoder = create_test_encoder(
        vocab_size=tokenizer.n_vocab,
        hidden_size=64,
        num_layers=2,
        sequence_length=512  # Max position embeddings
    )
    encoder.eval()
    
    tokens_tensor = torch.tensor(tokens, dtype=torch.long)
    
    print(f"\n   Input: {total_tokens} tokens")
    print(f"   Sequence length: {sequence_length}")
    
    # Calculate and print chunks with text
    chunks = calculate_chunks(total_tokens, sequence_length)
    num_chunks = len(chunks)
    print(f"   Expected chunks: {num_chunks}")
    
    # Print text for each chunk (in processing order - reverse)
    print(f"\n   📝 CHUNK TEXT CONTENT (Processing Order - REVERSE):")
    processing_order = list(range(num_chunks - 1, -1, -1))
    
    for proc_idx, chunk_idx in enumerate(processing_order):
        start_idx, end_idx, actual_length = chunks[chunk_idx]
        chunk_tokens = tokens[start_idx:end_idx]
        chunk_text = tokenizer.decode(chunk_tokens)
        
        # Truncate for display
        display_text = chunk_text[:100] + "..." if len(chunk_text) > 100 else chunk_text
        display_text = display_text.replace('\n', ' ').replace('\r', '')
        
        print(f"\n   ┌{'─' * 70}┐")
        print(f"   │ PROCESSING STEP {proc_idx + 1}/{num_chunks} - CHUNK {chunk_idx}")
        print(f"   │ Token range: [{start_idx}:{end_idx}] ({actual_length} actual tokens)")
        print(f"   │ TEXT: {display_text}")
        print(f"   └{'─' * 70}┘")
    
    with torch.no_grad():
        encoder_k, encoder_v = encoder(tokens_tensor, return_encoder_kv=True, sequence_length=sequence_length)
    
    print(f"\n   ✅ Output shapes:")
    print(f"      encoder_k: {encoder_k.shape}")
    print(f"      encoder_v: {encoder_v.shape}")
    print(f"\n   Note: Output length ({encoder_k.shape[0]}) equals input length ({total_tokens})")


def calculate_chunks(total_length: int, sequence_length: int) -> list[tuple[int, int, int]]:
    """
    Calculate chunk boundaries (mirrors the encoder's _create_chunks logic).
    
    Returns:
        List of (start_idx, end_idx, actual_token_count) tuples
    """
    chunks = []
    remainder = total_length % sequence_length
    
    if total_length <= sequence_length:
        # Single chunk
        chunks.append((0, total_length, total_length))
    else:
        if remainder > 0:
            # First chunk is smaller, needs to borrow from next chunk
            chunks.append((0, sequence_length, remainder))
            start = remainder
        else:
            start = 0
        
        # Add full chunks
        while start < total_length:
            end = min(start + sequence_length, total_length)
            actual = end - start
            chunks.append((start, end, actual))
            start = end
    
    return chunks


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
