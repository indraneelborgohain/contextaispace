#!/usr/bin/env python3
"""
test_encoder_decoder_integration.py - Integration tests for encoder-decoder architecture

Tests the full pipeline: encoder processes context -> decoder generates output with cross-attention
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F

from architecture.encoder import BidirectionalEncoder, EncoderConfig
from architecture.transformer import Transformer
from architecture.config import ModelConfig


def generate_tokens(batch_size, seq_len, vocab_size=1000):
    """Generate random token IDs for testing"""
    return torch.randint(1, vocab_size, (batch_size, seq_len), dtype=torch.long)


def create_test_models(vocab_size=1000, hidden_size=64, num_layers=2, 
                       encoder_seq_len=100, decoder_window=64):
    """Create matching encoder and decoder for testing"""
    encoder_config = EncoderConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        num_experts=4,
        experts_per_token=2,
        max_position_embeddings=encoder_seq_len * 5,
        dropout=0.0,
        use_moe=False,
        use_lsi_compression=False,
    )
    
    decoder_config = ModelConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_experts=4,
        experts_per_token=2,
        intermediate_size=128,
        sliding_window=decoder_window,
        initial_context_length=256,
        use_lsi_cross_attention=False,
        use_context_embedding=True,
        use_encoder_decoder_cross_attention=True,
        context_dim=hidden_size,
    )
    
    encoder = BidirectionalEncoder(encoder_config, device=torch.device('cpu'))
    decoder = Transformer(decoder_config, device=torch.device('cpu'))
    
    return encoder, decoder


class IntegrationTester:
    """Integration test suite for encoder-decoder architecture"""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.passed = 0
        self.failed = 0
        
    def log(self, message):
        if self.verbose:
            print(f"  {message}")
    
    def assert_true(self, condition, name):
        """Assert a condition is true"""
        if condition:
            self.passed += 1
            self.log(f"✓ {name}")
            return True
        else:
            self.failed += 1
            print(f"✗ FAILED: {name}")
            return False
    
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
    
    def test_basic_pipeline(self, encoder, decoder):
        """Test Case 1: Basic encoder -> decoder pipeline"""
        print("\nTest Case 1: Basic Encoder-Decoder Pipeline")
        print("-" * 60)
        
        # Generate context and query
        context_tokens = generate_tokens(1, 100)[0]
        query_tokens = generate_tokens(1, 20)[0]
        
        self.log(f"Context: {context_tokens.shape[0]} tokens")
        self.log(f"Query: {query_tokens.shape[0]} tokens")
        
        # Encode context
        encoder_k, encoder_v = encoder(context_tokens, return_encoder_kv=True)
        
        self.assert_shape(encoder_k, (100, encoder.config.hidden_size), "encoder_k")
        self.assert_shape(encoder_v, (100, encoder.config.hidden_size), "encoder_v")
        
        # Decode with cross-attention
        decoder.reset_context()
        logits = decoder(query_tokens, encoder_k=encoder_k, encoder_v=encoder_v)
        
        self.assert_shape(logits, (20, decoder.config.vocab_size), "decoder logits")
        
        self.log("Basic pipeline verified")
    
    def test_various_context_lengths(self, encoder, decoder):
        """Test Case 2: Various context lengths"""
        print("\nTest Case 2: Various Context Lengths")
        print("-" * 60)
        
        query_tokens = generate_tokens(1, 20)[0]
        
        for ctx_len in [50, 100, 210, 500]:
            self.log(f"Testing context length: {ctx_len}")
            
            context_tokens = generate_tokens(1, ctx_len)[0]
            
            # Encode
            encoder_k, encoder_v = encoder(context_tokens, return_encoder_kv=True, sequence_length=100)
            
            # Decode
            decoder.reset_context()
            logits = decoder(query_tokens, encoder_k=encoder_k, encoder_v=encoder_v)
            
            self.assert_shape(logits, (20, decoder.config.vocab_size), 
                            f"logits (ctx_len={ctx_len})")
    
    def test_various_query_lengths(self, encoder, decoder):
        """Test Case 3: Various query lengths"""
        print("\nTest Case 3: Various Query Lengths")
        print("-" * 60)
        
        context_tokens = generate_tokens(1, 100)[0]
        encoder_k, encoder_v = encoder(context_tokens, return_encoder_kv=True)
        
        for query_len in [10, 50, 100, 200]:
            self.log(f"Testing query length: {query_len}")
            
            query_tokens = generate_tokens(1, query_len)[0]
            
            decoder.reset_context()
            logits = decoder(query_tokens, encoder_k=encoder_k, encoder_v=encoder_v, 
                           max_seq_len=decoder.config.sliding_window)
            
            self.assert_shape(logits, (query_len, decoder.config.vocab_size),
                            f"logits (query_len={query_len})")
    
    def test_gradient_flow(self, encoder, decoder):
        """Test Case 4: Gradient flow through encoder-decoder"""
        print("\nTest Case 4: Gradient Flow Through Pipeline")
        print("-" * 60)
        
        encoder.train()
        decoder.train()
        
        context_tokens = generate_tokens(1, 100)[0]
        query_tokens = generate_tokens(1, 20)[0]
        targets = generate_tokens(1, 20)[0]
        
        self.log("Forward pass...")
        
        # Forward
        encoder_k, encoder_v = encoder(context_tokens, return_encoder_kv=True)
        decoder.reset_context()
        logits = decoder(query_tokens, encoder_k=encoder_k, encoder_v=encoder_v)
        
        # Compute loss
        loss = F.cross_entropy(
            logits.view(-1, decoder.config.vocab_size),
            targets.view(-1)
        )
        
        self.log(f"Loss: {loss.item():.4f}")
        
        # Backward
        loss.backward()
        
        # Check encoder gradients
        encoder_has_grads = False
        for name, param in encoder.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                encoder_has_grads = True
                break
        
        # Check decoder gradients
        decoder_has_grads = False
        for name, param in decoder.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                decoder_has_grads = True
                break
        
        self.assert_true(encoder_has_grads, "Encoder has gradients")
        self.assert_true(decoder_has_grads, "Decoder has gradients")
        
        encoder.zero_grad()
        decoder.zero_grad()
        encoder.eval()
        decoder.eval()
    
    def test_training_step(self, encoder, decoder):
        """Test Case 5: Full training step with optimizer"""
        print("\nTest Case 5: Full Training Step")
        print("-" * 60)
        
        encoder.train()
        decoder.train()
        
        # Create optimizer
        params = list(encoder.parameters()) + list(decoder.parameters())
        optimizer = torch.optim.AdamW(params, lr=1e-4)
        
        context_tokens = generate_tokens(1, 100)[0]
        query_tokens = generate_tokens(1, 20)[0]
        targets = generate_tokens(1, 20)[0]
        
        # Training step
        optimizer.zero_grad()
        
        encoder_k, encoder_v = encoder(context_tokens, return_encoder_kv=True)
        decoder.reset_context()
        logits = decoder(query_tokens, encoder_k=encoder_k, encoder_v=encoder_v)
        
        loss_before = F.cross_entropy(
            logits.view(-1, decoder.config.vocab_size),
            targets.view(-1)
        )
        
        loss_before.backward()
        optimizer.step()
        
        self.log(f"Loss before: {loss_before.item():.4f}")
        
        # Second forward pass to check loss changed
        optimizer.zero_grad()
        encoder_k, encoder_v = encoder(context_tokens, return_encoder_kv=True)
        decoder.reset_context()
        logits = decoder(query_tokens, encoder_k=encoder_k, encoder_v=encoder_v)
        
        loss_after = F.cross_entropy(
            logits.view(-1, decoder.config.vocab_size),
            targets.view(-1)
        )
        
        self.log(f"Loss after:  {loss_after.item():.4f}")
        
        # Parameters should have changed
        self.assert_true(loss_before.item() != loss_after.item(), "Loss changed after optimizer step")
        
        encoder.eval()
        decoder.eval()
    
    def test_batch_processing(self, encoder, decoder):
        """Test Case 6: Processing multiple examples"""
        print("\nTest Case 6: Batch Processing (Sequential)")
        print("-" * 60)
        
        batch_size = 4
        
        for i in range(batch_size):
            context_tokens = generate_tokens(1, 100 + i * 10)[0]
            query_tokens = generate_tokens(1, 20 + i * 5)[0]
            
            self.log(f"Example {i+1}: ctx={context_tokens.shape[0]}, query={query_tokens.shape[0]}")
            
            encoder_k, encoder_v = encoder(context_tokens, return_encoder_kv=True)
            decoder.reset_context()
            logits = decoder(query_tokens, encoder_k=encoder_k, encoder_v=encoder_v)
            
            expected_shape = (query_tokens.shape[0], decoder.config.vocab_size)
            self.assert_shape(logits, expected_shape, f"example {i+1} logits")
    
    def test_context_state_isolation(self, encoder, decoder):
        """Test Case 7: Decoder context isolation between examples"""
        print("\nTest Case 7: Context State Isolation")
        print("-" * 60)
        
        context1 = generate_tokens(1, 100)[0]
        query1 = generate_tokens(1, 20)[0]
        
        # Process first example
        encoder_k1, encoder_v1 = encoder(context1, return_encoder_kv=True)
        decoder.reset_context()
        logits1 = decoder(query1, encoder_k=encoder_k1, encoder_v=encoder_v1, update_context=True)
        context_state1 = decoder.context_state.clone()
        
        # Process second example (should have fresh context)
        context2 = generate_tokens(1, 100)[0]
        query2 = generate_tokens(1, 20)[0]
        
        encoder_k2, encoder_v2 = encoder(context2, return_encoder_kv=True)
        decoder.reset_context()  # Reset for new example
        logits2 = decoder(query2, encoder_k=encoder_k2, encoder_v=encoder_v2, update_context=True)
        context_state2 = decoder.context_state.clone()
        
        # Context states should be different (different inputs)
        different_contexts = not torch.allclose(context_state1, context_state2, rtol=1e-3)
        self.assert_true(different_contexts, "Context states isolated between examples")
        
        self.log("Context isolation verified")
    
    def test_long_context_long_query(self, encoder, decoder):
        """Test Case 8: Both encoder and decoder with chunking"""
        print("\nTest Case 8: Long Context + Long Query (Both Chunking)")
        print("-" * 60)
        
        # Long context (will chunk in encoder)
        context_tokens = generate_tokens(1, 300)[0]
        # Long query (will chunk in decoder)
        query_tokens = generate_tokens(1, 150)[0]
        
        self.log(f"Context: {context_tokens.shape[0]} tokens")
        self.log(f"Query: {query_tokens.shape[0]} tokens")
        
        # Encode with chunking
        encoder_k, encoder_v = encoder(context_tokens, return_encoder_kv=True, sequence_length=100)
        
        self.assert_shape(encoder_k, (300, encoder.config.hidden_size), "encoder_k (chunked)")
        self.assert_shape(encoder_v, (300, encoder.config.hidden_size), "encoder_v (chunked)")
        
        # Decode with chunking
        decoder.reset_context()
        logits = decoder(query_tokens, encoder_k=encoder_k, encoder_v=encoder_v, 
                        max_seq_len=decoder.config.sliding_window)
        
        self.assert_shape(logits, (150, decoder.config.vocab_size), "logits (chunked)")
        
        self.log("Both encoder and decoder chunking verified")
    
    def test_deterministic_pipeline(self, encoder, decoder):
        """Test Case 9: Deterministic end-to-end"""
        print("\nTest Case 9: Deterministic Pipeline")
        print("-" * 60)
        
        torch.manual_seed(42)
        context_tokens = generate_tokens(1, 100)[0]
        query_tokens = generate_tokens(1, 20)[0]
        
        with torch.no_grad():
            # First run
            encoder_k1, encoder_v1 = encoder(context_tokens, return_encoder_kv=True)
            decoder.reset_context()
            logits1 = decoder(query_tokens, encoder_k=encoder_k1, encoder_v=encoder_v1)
            
            # Second run
            encoder_k2, encoder_v2 = encoder(context_tokens, return_encoder_kv=True)
            decoder.reset_context()
            logits2 = decoder(query_tokens, encoder_k=encoder_k2, encoder_v=encoder_v2)
        
        encoder_match = torch.allclose(encoder_k1, encoder_k2, rtol=1e-5)
        decoder_match = torch.allclose(logits1, logits2, rtol=1e-5)
        
        self.assert_true(encoder_match, "Encoder output is deterministic")
        self.assert_true(decoder_match, "Decoder output is deterministic")
    
    def test_qa_scenario(self, encoder, decoder):
        """Test Case 10: Question-answering scenario simulation"""
        print("\nTest Case 10: Q&A Scenario Simulation")
        print("-" * 60)
        
        # Simulate: context passage -> question -> answer generation
        context = generate_tokens(1, 200)[0]  # Passage
        question = generate_tokens(1, 15)[0]   # Question
        
        self.log(f"Context (passage): {context.shape[0]} tokens")
        self.log(f"Question: {question.shape[0]} tokens")
        
        # Encode context
        encoder_k, encoder_v = encoder(context, return_encoder_kv=True, sequence_length=100)
        
        # Prepare decoder input (question + answer start)
        decoder_input = question
        
        # Generate answer tokens one by one
        decoder.reset_context()
        max_answer_len = 30
        
        generated_logits = []
        current_tokens = decoder_input
        
        for i in range(max_answer_len):
            # Get logits for current sequence
            logits = decoder(current_tokens, encoder_k=encoder_k, encoder_v=encoder_v, 
                           update_context=False)
            
            # Get next token (greedy)
            next_token_id = torch.argmax(logits[-1], dim=-1).unsqueeze(0)
            
            # Add to sequence
            current_tokens = torch.cat([current_tokens, next_token_id])
            generated_logits.append(logits[-1].unsqueeze(0))
            
            # Stop after a few tokens for testing
            if i >= 5:
                break
        
        self.log(f"Generated {len(generated_logits)} answer tokens")
        self.assert_true(len(generated_logits) > 0, "Answer generation working")
        
        # Verify final sequence shape
        self.log(f"Final sequence length: {current_tokens.shape[0]}")
        self.assert_true(current_tokens.shape[0] == question.shape[0] + len(generated_logits),
                        "Generated sequence has correct length")
    
    def test_loss_masking(self, encoder, decoder):
        """Test Case 11: Loss computation with masking (Q&A style)"""
        print("\nTest Case 11: Loss Computation with Masking")
        print("-" * 60)
        
        context_tokens = generate_tokens(1, 100)[0]
        
        # Simulate: question (10 tokens) + answer (10 tokens)
        question = generate_tokens(1, 10)[0]
        answer = generate_tokens(1, 10)[0]
        decoder_input = torch.cat([question, answer])
        
        # Target: mask question tokens, compute loss only on answer
        targets = torch.cat([
            torch.full((10,), -100, dtype=torch.long),  # Mask question
            answer  # Answer tokens
        ])
        
        self.log(f"Question tokens: 10 (masked)")
        self.log(f"Answer tokens: 10 (for loss)")
        
        # Forward pass
        encoder_k, encoder_v = encoder(context_tokens, return_encoder_kv=True)
        decoder.reset_context()
        logits = decoder(decoder_input, encoder_k=encoder_k, encoder_v=encoder_v)
        
        # Compute loss (should ignore masked tokens)
        loss = F.cross_entropy(
            logits.view(-1, decoder.config.vocab_size),
            targets.view(-1),
            ignore_index=-100
        )
        
        self.log(f"Loss: {loss.item():.4f}")
        self.assert_true(not torch.isnan(loss) and not torch.isinf(loss), 
                        "Loss is valid with masking")
    
    def run_all_tests(self):
        """Run all integration tests"""
        print("=" * 60)
        print("ENCODER-DECODER INTEGRATION TEST SUITE")
        print("=" * 60)
        
        encoder, decoder = create_test_models()
        encoder.eval()
        decoder.eval()
        
        try:
            self.test_basic_pipeline(encoder, decoder)
            self.test_various_context_lengths(encoder, decoder)
            self.test_various_query_lengths(encoder, decoder)
            
            # Training tests
            self.test_gradient_flow(encoder, decoder)
            self.test_training_step(encoder, decoder)
            
            # Batch and isolation tests
            self.test_batch_processing(encoder, decoder)
            self.test_context_state_isolation(encoder, decoder)
            
            # Advanced scenarios
            self.test_long_context_long_query(encoder, decoder)
            self.test_deterministic_pipeline(encoder, decoder)
            self.test_qa_scenario(encoder, decoder)
            self.test_loss_masking(encoder, decoder)
            
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
            print("\n✓ ALL INTEGRATION TESTS PASSED!")
            return 0
        else:
            print(f"\n✗ {self.failed} TESTS FAILED")
            return 1


def main():
    """Run the integration test suite"""
    tester = IntegrationTester(verbose=True)
    exit_code = tester.run_all_tests()
    return exit_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
