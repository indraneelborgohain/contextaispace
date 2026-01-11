#!/usr/bin/env python3
"""
test_decoder_new.py - Comprehensive test suite for Transformer decoder with cross-attention

Tests the decoder's forward pass, cross-attention to encoder, and context management.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn

from architecture.transformer import Transformer
from architecture.config import ModelConfig


def generate_tokens(batch_size, seq_len, vocab_size=1000):
    """Generate random token IDs for testing"""
    return torch.randint(1, vocab_size, (batch_size, seq_len), dtype=torch.long)


def create_test_decoder(vocab_size=1000, hidden_size=64, num_layers=2, sliding_window=128):
    """Create a small decoder for testing"""
    config = ModelConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_experts=4,
        experts_per_token=2,
        intermediate_size=128,
        sliding_window=sliding_window,
        initial_context_length=256,
        use_lsi_cross_attention=False,
        use_context_embedding=True,
        use_encoder_decoder_cross_attention=True,
    )
    return Transformer(config, device=torch.device('cpu'))


class DecoderTester:
    """Test suite for decoder forward method"""
    
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
    
    def test_basic_forward(self, decoder):
        """Test Case 1: Basic forward pass without encoder"""
        print("\nTest Case 1: Basic Forward Pass (No Encoder)")
        print("-" * 60)
        
        tokens = generate_tokens(1, 50)[0]
        self.log(f"Input: {tokens.shape[0]} tokens")
        
        decoder.reset_context()
        logits = decoder(tokens, update_context=True)
        
        # Should output logits for each token
        self.assert_shape(logits, (50, decoder.config.vocab_size), "logits")
        
        self.log("Basic forward pass verified")
    
    def test_forward_with_encoder_kv(self, decoder):
        """Test Case 2: Forward pass with encoder cross-attention"""
        print("\nTest Case 2: Forward with Encoder Cross-Attention")
        print("-" * 60)
        
        tokens = generate_tokens(1, 50)[0]
        
        # Create mock encoder K, V
        enc_len = 100
        hidden_size = decoder.config.hidden_size
        encoder_k = torch.randn(enc_len, hidden_size)
        encoder_v = torch.randn(enc_len, hidden_size)
        
        self.log(f"Input tokens: {tokens.shape[0]}")
        self.log(f"Encoder K: {encoder_k.shape}")
        self.log(f"Encoder V: {encoder_v.shape}")
        
        decoder.reset_context()
        logits = decoder(tokens, encoder_k=encoder_k, encoder_v=encoder_v, update_context=True)
        
        self.assert_shape(logits, (50, decoder.config.vocab_size), "logits with cross-attention")
        
        self.log("Forward with cross-attention verified")
    
    def test_context_state_management(self, decoder):
        """Test Case 3: Context state updates"""
        print("\nTest Case 3: Context State Management")
        print("-" * 60)
        
        # Reset context
        decoder.reset_context()
        initial_context = decoder.context_state.clone()
        
        self.log(f"Initial context: {initial_context.abs().sum().item():.6f}")
        
        # Process some tokens
        tokens1 = generate_tokens(1, 20)[0]
        logits1 = decoder(tokens1, update_context=True)
        context_after_1 = decoder.context_state.clone()
        
        self.log(f"Context after sequence 1: {context_after_1.abs().sum().item():.6f}")
        
        # Process more tokens (context should update)
        tokens2 = generate_tokens(1, 20)[0]
        logits2 = decoder(tokens2, update_context=True)
        context_after_2 = decoder.context_state.clone()
        
        self.log(f"Context after sequence 2: {context_after_2.abs().sum().item():.6f}")
        
        # Check that context changed
        context_changed = not torch.allclose(initial_context, context_after_1, rtol=1e-5)
        context_updated = not torch.allclose(context_after_1, context_after_2, rtol=1e-5)
        
        if context_changed and context_updated:
            self.passed += 1
            self.log("✓ Context state updates correctly")
        else:
            self.failed += 1
            print("✗ FAILED: Context state not updating!")
        
        # Reset and verify
        decoder.reset_context()
        context_reset = decoder.context_state.clone()
        
        if torch.allclose(context_reset, initial_context, rtol=1e-5):
            self.passed += 1
            self.log("✓ Context reset works correctly")
        else:
            self.failed += 1
            print("✗ FAILED: Context not reset properly!")
    
    def test_chunking_short_sequence(self, decoder):
        """Test Case 4: Short sequence (no chunking)"""
        print("\nTest Case 4: Short Sequence Processing")
        print("-" * 60)
        
        # Sequence shorter than sliding window
        tokens = generate_tokens(1, 50)[0]
        self.log(f"Input: {tokens.shape[0]} tokens (sliding_window={decoder.config.sliding_window})")
        
        decoder.reset_context()
        logits = decoder(tokens, update_context=True)
        
        self.assert_shape(logits, (50, decoder.config.vocab_size), "logits (no chunking)")
        self.log("No chunking needed for short sequence")
    
    def test_chunking_long_sequence(self, decoder):
        """Test Case 5: Long sequence requiring chunking"""
        print("\nTest Case 5: Long Sequence with Chunking")
        print("-" * 60)
        
        # Sequence longer than sliding window
        sliding_window = decoder.config.sliding_window
        tokens = generate_tokens(1, sliding_window * 2 + 10)[0]  # e.g., 266 tokens
        
        self.log(f"Input: {tokens.shape[0]} tokens (sliding_window={sliding_window})")
        self.log(f"Expected chunks: ~{tokens.shape[0] // sliding_window + 1}")
        
        decoder.reset_context()
        logits = decoder(tokens, update_context=True, max_seq_len=sliding_window)
        
        # Should return logits for all tokens
        self.assert_shape(logits, (tokens.shape[0], decoder.config.vocab_size), "logits (with chunking)")
        self.log("Chunking processed correctly")
    
    def test_cross_attention_with_chunking(self, decoder):
        """Test Case 6: Cross-attention with chunking"""
        print("\nTest Case 6: Cross-Attention with Chunking")
        print("-" * 60)
        
        sliding_window = decoder.config.sliding_window
        tokens = generate_tokens(1, sliding_window + 50)[0]
        
        # Create encoder K, V
        hidden_size = decoder.config.hidden_size
        encoder_k = torch.randn(200, hidden_size)
        encoder_v = torch.randn(200, hidden_size)
        
        self.log(f"Input tokens: {tokens.shape[0]}")
        self.log(f"Encoder K, V: {encoder_k.shape}")
        
        decoder.reset_context()
        logits = decoder(tokens, encoder_k=encoder_k, encoder_v=encoder_v, 
                        update_context=True, max_seq_len=sliding_window)
        
        self.assert_shape(logits, (tokens.shape[0], decoder.config.vocab_size), 
                         "logits (chunking + cross-attention)")
        self.log("Cross-attention with chunking verified")
    
    def test_no_context_update(self, decoder):
        """Test Case 7: Forward without updating context"""
        print("\nTest Case 7: Forward Without Context Update")
        print("-" * 60)
        
        decoder.reset_context()
        initial_context = decoder.context_state.clone()
        
        tokens = generate_tokens(1, 30)[0]
        logits = decoder(tokens, update_context=False)
        
        final_context = decoder.context_state.clone()
        
        # Context should not have changed
        if torch.allclose(initial_context, final_context, rtol=1e-5):
            self.passed += 1
            self.log("✓ Context not updated when update_context=False")
        else:
            self.failed += 1
            print("✗ FAILED: Context was updated despite update_context=False!")
    
    def test_gradient_flow(self, decoder):
        """Test Case 8: Gradient flow"""
        print("\nTest Case 8: Gradient Flow")
        print("-" * 60)
        
        tokens = generate_tokens(1, 50)[0]
        
        decoder.reset_context()
        decoder.train()
        
        logits = decoder(tokens, update_context=True)
        
        # Compute a simple loss
        loss = logits.sum()
        loss.backward()
        
        # Check gradients
        has_grads = False
        for name, param in decoder.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grads = True
                break
        
        if has_grads:
            self.passed += 1
            self.log("✓ Gradients flow correctly")
        else:
            self.failed += 1
            print("✗ FAILED: No gradients detected!")
        
        decoder.zero_grad()
        decoder.eval()
    
    def test_determinism(self, decoder):
        """Test Case 9: Deterministic output"""
        print("\nTest Case 9: Deterministic Output")
        print("-" * 60)
        
        torch.manual_seed(42)
        tokens = generate_tokens(1, 50)[0]
        
        decoder.eval()
        with torch.no_grad():
            decoder.reset_context()
            logits1 = decoder(tokens, update_context=True)
            
            decoder.reset_context()
            logits2 = decoder(tokens, update_context=True)
        
        if torch.allclose(logits1, logits2, rtol=1e-5, atol=1e-7):
            self.passed += 1
            self.log("✓ Decoder output is deterministic")
        else:
            self.failed += 1
            print("✗ FAILED: Non-deterministic output!")
            print(f"  Max difference: {(logits1 - logits2).abs().max().item()}")
    
    def test_different_encoder_lengths(self, decoder):
        """Test Case 10: Different encoder K,V lengths"""
        print("\nTest Case 10: Various Encoder K,V Lengths")
        print("-" * 60)
        
        tokens = generate_tokens(1, 50)[0]
        hidden_size = decoder.config.hidden_size
        
        for enc_len in [50, 100, 200, 500]:
            encoder_k = torch.randn(enc_len, hidden_size)
            encoder_v = torch.randn(enc_len, hidden_size)
            
            self.log(f"Testing with encoder length: {enc_len}")
            
            decoder.reset_context()
            logits = decoder(tokens, encoder_k=encoder_k, encoder_v=encoder_v)
            
            self.assert_shape(logits, (50, decoder.config.vocab_size), 
                            f"logits (enc_len={enc_len})")
    
    def run_all_tests(self):
        """Run all test cases"""
        print("=" * 60)
        print("DECODER COMPREHENSIVE TEST SUITE")
        print("=" * 60)
        
        decoder = create_test_decoder()
        decoder.eval()
        
        try:
            self.test_basic_forward(decoder)
            self.test_forward_with_encoder_kv(decoder)
            self.test_context_state_management(decoder)
            self.test_chunking_short_sequence(decoder)
            self.test_chunking_long_sequence(decoder)
            self.test_cross_attention_with_chunking(decoder)
            self.test_no_context_update(decoder)
            
            decoder.train()
            self.test_gradient_flow(decoder)
            
            decoder.eval()
            self.test_determinism(decoder)
            self.test_different_encoder_lengths(decoder)
            
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
    tester = DecoderTester(verbose=True)
    exit_code = tester.run_all_tests()
    return exit_code


def test_decoder_chunking_visualization():
    """
    Test that visualizes the decoder chunking logic by printing each chunk.
    Shows input tokens and expected output (next token prediction) per chunk.
    """
    print("=" * 80)
    print("DECODER CHUNKING VISUALIZATION TEST")
    print("=" * 80)
    
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from architecture.tokenizer import get_tokenizer
    
    # Get tokenizer
    tokenizer = get_tokenizer()
    
    # Create an answer sequence to decode
    answer_text = """<A>Transformers revolutionized NLP through their attention mechanisms. 
    The self-attention layer allows each token to attend to all other tokens in the sequence.
    This enables the model to capture long-range dependencies that were difficult for RNNs.
    The encoder-decoder architecture uses cross-attention to connect input and output.
    Multi-head attention allows the model to focus on different aspects simultaneously.
    Position embeddings give the model information about token order in the sequence.
    Layer normalization and residual connections help with training stability.
    The feed-forward network processes each position independently after attention.
    Pre-training on large corpora followed by fine-tuning has proven very effective."""
    
    # Tokenize the answer
    tokens = tokenizer.encode(answer_text)
    total_tokens = len(tokens)
    
    print(f"\n📝 DECODER INPUT DETAILS:")
    print(f"   Total tokens: {total_tokens}")
    print(f"   First 50 chars: {answer_text[:50]}...")
    print(f"   Last 50 chars: ...{answer_text[-50:]}")
    
    # Test with different sliding window sizes
    for sliding_window in [64, 128, 256]:
        print(f"\n{'─' * 80}")
        print(f"🔧 SLIDING WINDOW (max_seq_len): {sliding_window}")
        print(f"{'─' * 80}")
        
        # Calculate chunks (decoder uses forward order, unlike encoder)
        num_chunks = (total_tokens + sliding_window - 1) // sliding_window
        
        print(f"\n📦 NUMBER OF CHUNKS: {num_chunks}")
        print(f"   Processing order: FORWARD (first chunk first)\n")
        
        # Print each chunk with input/output details
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * sliding_window
            end_idx = min(start_idx + sliding_window, total_tokens)
            chunk_length = end_idx - start_idx
            
            # Get chunk tokens
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = tokenizer.decode(chunk_tokens)
            
            # INPUT: The tokens in this chunk
            input_token_ids = chunk_tokens
            input_text = chunk_text
            
            # EXPECTED OUTPUT: For each input token, predict next token
            # Output shape is (chunk_length, vocab_size) -> argmax gives predicted next tokens
            # The target for training would be tokens[start_idx+1:end_idx+1]
            if end_idx < total_tokens:
                target_tokens = tokens[start_idx + 1:end_idx + 1]
                target_text = tokenizer.decode(target_tokens)
            else:
                # Last chunk: target is tokens[start_idx+1:end_idx] + <end>
                target_tokens = tokens[start_idx + 1:end_idx] if end_idx > start_idx + 1 else []
                target_text = tokenizer.decode(target_tokens) if target_tokens else "(end of sequence)"
            
            # Truncate text for display
            display_input = input_text[:80] + "..." if len(input_text) > 80 else input_text
            display_target = target_text[:80] + "..." if len(target_text) > 80 else target_text
            
            # Determine chunk type
            if chunk_idx == 0:
                chunk_type = "FIRST (uses learnable <start> token)"
                context_source = "Learnable start_token_embedding"
            else:
                chunk_type = "CONTINUATION (uses previous context)"
                context_source = f"context_state from chunk {chunk_idx - 1}'s last token"
            
            print(f"   ┌{'─' * 70}┐")
            print(f"   │ CHUNK {chunk_idx} ({chunk_type})")
            print(f"   │ Token range: [{start_idx}:{end_idx}] ({chunk_length} tokens)")
            if chunk_length < sliding_window:
                print(f"   │ ⚠️  Partial chunk (padded to {sliding_window} for processing)")
            print(f"   │")
            print(f"   │ 🔹 CONTEXT SOURCE: {context_source}")
            print(f"   │")
            print(f"   │ 📥 INPUT TOKENS: {input_token_ids[:10]}{'...' if len(input_token_ids) > 10 else ''}")
            print(f"   │    Input text: {display_input}")
            print(f"   │")
            print(f"   │ 📤 EXPECTED OUTPUT (next token prediction targets):")
            if target_tokens:
                print(f"   │    Target tokens: {target_tokens[:10]}{'...' if len(target_tokens) > 10 else ''}")
            print(f"   │    Target text: {display_target}")
            print(f"   │")
            print(f"   │ 🔸 OUTPUT SHAPE: ({chunk_length}, vocab_size)")
            print(f"   │    Logits[i] predicts token at position {start_idx}+i+1")
            print(f"   └{'─' * 70}┘")
            print()
        
        # Print context flow
        print(f"   🔄 CONTEXT STATE FLOW:")
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * sliding_window
            end_idx = min(start_idx + sliding_window, total_tokens)
            
            if chunk_idx == 0:
                print(f"      Chunk {chunk_idx}: start_token_embedding → process → "
                      f"context_state = hidden_state[{end_idx - 1}]")
            else:
                print(f"      Chunk {chunk_idx}: context_state[{chunk_idx-1}] → process → "
                      f"context_state = hidden_state[{end_idx - 1}]")
    
    # Test with actual decoder
    print(f"\n{'=' * 80}")
    print("🧪 ACTUAL DECODER TEST WITH CHUNKING")
    print(f"{'=' * 80}")
    
    sliding_window = 64
    decoder = create_test_decoder(
        vocab_size=tokenizer.n_vocab,
        hidden_size=64,
        num_layers=2,
        sliding_window=sliding_window
    )
    decoder.eval()
    
    tokens_tensor = torch.tensor(tokens, dtype=torch.long)
    
    print(f"\n   Input: {total_tokens} tokens")
    print(f"   Sliding window: {sliding_window}")
    print(f"   Expected chunks: {(total_tokens + sliding_window - 1) // sliding_window}")
    
    # Also test with encoder K, V
    encoder_len = 100
    encoder_k = torch.randn(encoder_len, 64)
    encoder_v = torch.randn(encoder_len, 64)
    
    print(f"   Encoder K shape: {encoder_k.shape}")
    print(f"   Encoder V shape: {encoder_v.shape}")
    
    with torch.no_grad():
        decoder.reset_context()
        logits = decoder(tokens_tensor, encoder_k=encoder_k, encoder_v=encoder_v)
    
    print(f"\n   ✅ Output shapes:")
    print(f"      logits: {logits.shape}")
    print(f"\n   Note: Output length ({logits.shape[0]}) equals input length ({total_tokens})")
    
    # Show what happens at each position
    print(f"\n   📊 PREDICTION MAPPING (first 10 positions):")
    for i in range(min(10, total_tokens)):
        input_token = tokens[i]
        if i + 1 < total_tokens:
            target_token = tokens[i + 1]
            predicted_token = logits[i].argmax().item()
            print(f"      Position {i}: Input '{tokenizer.decode([input_token])}' "
                  f"→ Predict '{tokenizer.decode([predicted_token])}' "
                  f"(Target: '{tokenizer.decode([target_token])}')")
        else:
            print(f"      Position {i}: Input '{tokenizer.decode([input_token])}' → (end of sequence)")
    
    print(f"\n   ✅ Decoder chunking visualization complete!")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--chunking":
        test_decoder_chunking_visualization()
    else:
        exit_code = main()
        sys.exit(exit_code)
