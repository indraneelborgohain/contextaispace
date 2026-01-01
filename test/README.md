# Test Suite

This directory contains comprehensive tests for the encoder-decoder architecture.

## Test Files

### 1. `test_components.py`
Tests individual components in isolation:
- **RMSNorm**: Layer normalization
- **RotaryEmbedding**: Position embeddings
- **ContextAttentionBlock**: Context-aware attention
- **CrossAttentionLayer**: Encoder-decoder cross-attention
- **Encoder Bidirectionality**: Verifies non-causal attention
- **Decoder Causality**: Verifies causal masking

### 2. `test_encoder_decoder.py`
Integration tests for encoder-decoder pipeline:
- Short context (64 tokens)
- Medium context (128 tokens)
- Long context (512+ tokens)
- Variable Q&A lengths (20-200 tokens)
- SVD compression quality
- Cross-attention effect verification
- Gradient flow validation

### 3. `test_squad_workflow.py`
End-to-end SQuAD-like Q&A workflow:
- Context encoding (128, 512, 1024 tokens)
- Question + Answer decoding
- Loss computation (only on answer tokens)
- Comparison with/without encoder
- Multiple test cases with various lengths

### 4. `run_all_tests.py`
Master test runner that executes all test suites and provides a summary.

## Running Tests

### Run all tests:
```bash
python test/run_all_tests.py
```

### Run individual test suites:
```bash
# Component tests
python test/test_components.py

# Encoder-decoder integration tests
python test/test_encoder_decoder.py

# SQuAD workflow tests
python test/test_squad_workflow.py
```

### Run with pytest (if installed):
```bash
pytest test/test_encoder_decoder.py -v
```

## Test Coverage

### Context Lengths Tested
- **Short**: 64-128 tokens
- **Medium**: 128-256 tokens
- **Long**: 512-1024 tokens

### Q&A Lengths Tested
- **Short**: 10-30 tokens
- **Medium**: 30-100 tokens
- **Long**: 100-200 tokens

### Edge Cases
- Empty sequences
- Single token
- Maximum context length
- Mismatched encoder/decoder sizes
- Gradient flow verification
- NaN/Inf detection

## Expected Output

All tests should pass with output similar to:
```
================================================================================
RUNNING COMPLETE TEST SUITE
================================================================================

✓ Component Tests completed successfully
✓ Encoder-Decoder Integration Tests completed successfully
✓ SQuAD Workflow Tests completed successfully

================================================================================
TEST SUMMARY
================================================================================
✓ Component Tests: PASSED
✓ Encoder-Decoder Integration Tests: PASSED
✓ SQuAD Workflow Tests: PASSED

Total: 3 tests
Passed: 3
Failed: 0

================================================================================
ALL TESTS PASSED ✓✓✓
================================================================================
```

## Debugging Failed Tests

If tests fail, check:
1. **Device compatibility**: CUDA vs CPU
2. **Memory**: Reduce batch sizes or sequence lengths
3. **Dependencies**: Ensure torch is properly installed
4. **Architecture changes**: Update tests if model architecture changed

## Adding New Tests

To add new test cases:
1. Create a new test function in the appropriate file
2. Follow the existing pattern with clear print statements
3. Include assertions for expected behavior
4. Add edge case handling
5. Update this README with test description

python train_encoder_decoder_stories.py \
  --model_size medium \
  --pretrained_decoder_path "model_context/checkpoint_5000.pt" \
  --use_lsi_compression \
  --encoder_chunk_size 256