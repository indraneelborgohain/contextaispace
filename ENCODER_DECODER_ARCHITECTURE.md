# Encoder-Decoder Cross-Attention with SVD Compression

## Architecture Overview

This implements a sophisticated encoder-decoder architecture with SVD-based compression for efficient cross-attention.

## Workflow (256 tokens, window=128)

### Encoder Processing
1. **Chunk 1** (tokens 0-127): Process → output (128, hidden_dim)
2. **Chunk 2** (tokens 128-255): Process → output (128, hidden_dim)  
3. **Stack**: Concatenate all chunk outputs → (256, hidden_dim)
4. **SVD Compression**: Reduce from 256 to 128 dimensions
   - Apply SVD to extract top 128 singular vectors
   - Result: K_enc, V_enc → (128, hidden_dim) each
5. **Output**: Compressed K, V for decoder cross-attention

### Decoder Processing
1. **Chunk 1** (tokens 0-127):
   - Embed tokens + prepend start/context token
   - **Self-attention** (causal masked)
   - **Cross-attention**: Q from chunk, K=K_enc, V=V_enc (SVD compressed)
   - MLP
   - Extract last token context → context_1

2. **Chunk 2** (tokens 128-255):
   - Prepend context_1 from previous chunk
   - **Self-attention** (causal masked)
   - **Cross-attention**: Q from chunk, K=K_enc, V=V_enc (same compressed K, V)
   - MLP
   - Extract last token context → context_2

## Key Components

### 1. Encoder ([architecture/encoder.py](architecture/encoder.py))

**New Methods:**
- `forward(..., return_compressed_kv=True)` - Returns SVD-compressed K, V
- `_forward_with_svd_compression()` - Processes chunks and applies SVD
- `_compress_with_svd()` - SVD compression to target dimensions

**Usage:**
```python
encoder_k, encoder_v = encoder(
    context_tokens,
    return_compressed_kv=True,
    chunk_size=128
)
# encoder_k, encoder_v: (128, hidden_size)
```

### 2. Decoder ([architecture/gptoss_context.py](architecture/gptoss_context.py))

**New Components:**
- `CrossAttentionLayer` - Cross-attention module (Q from decoder, K/V from encoder)
- Updated `TransformerBlock` - Now includes optional cross-attention after self-attention
- Updated `forward()` - Accepts `encoder_k`, `encoder_v` parameters

**Cross-Attention Flow:**
```python
# In each decoder layer (TransformerBlock):
x = self.attn(x)                                    # Self-attention (causal)
x = self.cross_attn(x, encoder_k, encoder_v)        # Cross-attention (bidirectional to encoder)
x = self.mlp(x)                                     # MLP
```

**Usage:**
```python
logits = decoder(
    question_answer_tokens,
    encoder_k=encoder_k,
    encoder_v=encoder_v,
    max_seq_len=128
)
```

### 3. Configuration

**Enable encoder-decoder cross-attention:**
```python
config = ModelConfig(
    use_encoder_decoder_cross_attention=True,  # Enable cross-attention
    use_lsi_cross_attention=False,             # Optional: can use both
    sliding_window=128
)
```

## SQuAD Training Setup

### Data Format
```
Encoder input: Context passage
Decoder input: <question> Question text? <answer> Answer text
```

### Loss Computation
- Compute loss ONLY on answer tokens
- Question tokens: `ignore_index=-100`

### Example
```python
# Encode context
encoder_k, encoder_v = encoder(context_tokens, return_compressed_kv=True)

# Decode question + answer
logits = decoder(
    question_answer_tokens,
    encoder_k=encoder_k,
    encoder_v=encoder_v
)

# Loss only on answer positions
loss = F.cross_entropy(
    logits[answer_start:].view(-1, vocab_size),
    targets[answer_start:].view(-1),
    ignore_index=-100
)
```

## Testing

Run the test script:
```bash
python test_encoder_decoder.py
```

Tests:
1. Long sequences with chunking (256 encoder, 200 decoder tokens)
2. Short sequences without chunking (64 tokens)
3. Cross-attention effect verification

## Architecture Advantages

1. **Memory Efficiency**: SVD reduces encoder outputs from `num_chunks × window` to `window`
2. **Information Preservation**: SVD keeps most important singular vectors
3. **Consistent Cross-Attention**: Same compressed K, V used across all decoder chunks
4. **Scalability**: Can handle arbitrary length sequences via chunking
5. **Context Propagation**: Context flows between chunks via prepended embeddings

## File Changes

- [architecture/encoder.py](architecture/encoder.py) - Added SVD compression
- [architecture/gptoss_context.py](architecture/gptoss_context.py) - Added CrossAttentionLayer
- [test_encoder_decoder.py](test_encoder_decoder.py) - Updated tests

## Next Steps

1. Create SQuAD data loader
2. Add special tokens (`<question>`, `<answer>`)
3. Implement training script with proper loss masking
4. Fine-tune hyperparameters (SVD components, window size)
