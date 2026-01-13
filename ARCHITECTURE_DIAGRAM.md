# Architecture Diagram: Hybrid Encoder-Decoder with BERT + GPT-OSS

## Full System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ENCODER-DECODER ARCHITECTURE                         │
└─────────────────────────────────────────────────────────────────────────────┘

Input Text (Context + Question)
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ENCODER SIDE                                    │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │  Token Embedding                    ← BERT weights (if enabled)    │    │
│  └──────────────────────┬─────────────────────────────────────────────┘    │
│                         │                                                    │
│                         ▼                                                    │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │  Bidirectional Transformer Blocks (12 layers)                      │    │
│  │  ┌──────────────────────────────────────────────────────────┐     │    │
│  │  │  Multi-Head Self-Attention   ← BERT weights (if enabled) │     │    │
│  │  │  Feed-Forward Network (MoE)  ← BERT weights (if enabled) │     │    │
│  │  │  Layer Normalization         ← BERT weights (if enabled) │     │    │
│  │  └──────────────────────────────────────────────────────────┘     │    │
│  │  (Repeats for N layers)                                            │    │
│  └──────────────────────┬─────────────────────────────────────────────┘    │
│                         │                                                    │
│                         ▼                                                    │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │  Split at <SEP> Token                                             │    │
│  │  - Tokens BEFORE <SEP> = Context                                  │    │
│  │  - Tokens AFTER <SEP> = Question                                  │    │
│  └──────────┬─────────────────────────────────────────┬───────────────┘    │
│             │                                          │                     │
│             ▼ (Context)                                ▼ (Question)          │
│  ┌─────────────────────────┐              ┌──────────────────────────┐     │
│  │ Process Context Chunks  │              │ Process Question Chunks  │     │
│  │ → Extract K, V          │              │ → Extract Q              │     │
│  └───────────┬─────────────┘              └──────────┬───────────────┘     │
│              │                                        │                     │
│              ▼                                        │                     │
│  ┌────────────────────────────────────────────┐      │                     │
│  │  SVD Compression      ← YOUR CUSTOM ✨     │      │                     │
│  │  - Stack all context K, V                 │      │                     │
│  │  - Compress to fixed length (128/256)     │      │                     │
│  └──────────────────┬─────────────────────────┘      │                     │
│                     │                                 │                     │
│                     │ Compressed K, V                 │ Q from question     │
│                     └───────────┬─────────────────────┘                     │
│                                 ▼                                           │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │  Final Cross-Attention         ← YOUR CUSTOM LAYER ✨             │    │
│  │  Q (from question) attends to compressed K,V (from context)        │    │
│  └──────────────────────┬─────────────────────────────────────────────┘    │
│                         │                                                    │
│                         ▼                                                    │
│                  Compressed K, V                                             │
│               (fixed length: 128/256)                                        │
└─────────────────────────┼───────────────────────────────────────────────────┘
                          │
                          │ Cross-Attention Keys & Values
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DECODER SIDE                                    │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │  Token Embedding                    ← GPT-OSS weights (if enabled) │    │
│  └──────────────────────┬─────────────────────────────────────────────┘    │
│                         │                                                    │
│                         ▼                                                    │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │  Context State (previous output)    ← YOUR CUSTOM FEATURE ✨       │    │
│  │  - Feeds back into decoder                                         │    │
│  └──────────────────────┬─────────────────────────────────────────────┘    │
│                         │                                                    │
│                         ▼                                                    │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │  Transformer Blocks (12 layers)                                    │    │
│  │  ┌──────────────────────────────────────────────────────────┐     │    │
│  │  │  Causal Self-Attention       ← GPT-OSS weights           │     │    │
│  │  │  (with sliding window)       ← GPT-OSS weights           │     │    │
│  │  └──────────────────────────────────────────────────────────┘     │    │
│  │  ┌──────────────────────────────────────────────────────────┐     │    │
│  │  │  Encoder-Decoder Cross-Attn  ← YOUR CUSTOM LAYER ✨      │     │    │
│  │  │  (attends to encoder K, V)                               │     │    │
│  │  └──────────────────────────────────────────────────────────┘     │    │
│  │  ┌──────────────────────────────────────────────────────────┐     │    │
│  │  │  Context Projection          ← YOUR CUSTOM LAYER ✨      │     │    │
│  │  │  Feed-Forward (MoE)          ← GPT-OSS weights           │     │    │
│  │  └──────────────────────────────────────────────────────────┘     │    │
│  │  (Repeats for N layers)                                            │    │
│  └──────────────────────┬─────────────────────────────────────────────┘    │
│                         │                                                    │
│                         ▼                                                    │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │  Output Projection (LM Head)        ← GPT-OSS weights (if enabled) │    │
│  └──────────────────────┬─────────────────────────────────────────────┘    │
│                         │                                                    │
│                         ▼                                                    │
│                   Generated Tokens                                           │
│                   (Answer to Question)                                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Weight Sources

```
┌────────────────────┬──────────────────────┬────────────────────────────┐
│ Component          │ Loaded From          │ Learning Rate              │
├────────────────────┼──────────────────────┼────────────────────────────┤
│ Encoder Embedding  │ BERT (optional)      │ 1e-6 (very low)           │
│ Encoder Attention  │ BERT (optional)      │ 1e-6 (very low)           │
│ Encoder FFN        │ BERT (optional)      │ 1e-6 (very low)           │
│ Chunk Cross-Attn   │ YOUR TRAINED MODEL   │ 3e-4 (normal)             │
│ SVD Compression    │ YOUR TRAINED MODEL   │ 3e-4 (normal)             │
├────────────────────┼──────────────────────┼────────────────────────────┤
│ Decoder Embedding  │ GPT-OSS (optional)   │ 1e-6 (very low)           │
│ Decoder Attention  │ GPT-OSS (optional)   │ 1e-6 (very low)           │
│ Decoder MoE        │ GPT-OSS (optional)   │ 1e-6 (very low)           │
│ Context Projection │ YOUR TRAINED MODEL   │ 3e-4 (normal)             │
│ Cross-Attention    │ YOUR TRAINED MODEL   │ 3e-4 (normal)             │
└────────────────────┴──────────────────────┴────────────────────────────┘
```

## Data Flow Example

```
Question: "What is the capital of France?"
Context:  "France is a country in Europe. Paris is its capital city."

┌─────────────────────────────────────────────────────────────────┐
│ Step 1: Encode                                                  │
│ ┌─────────────────────────────────────────────────────────────┐│
│ │ Input: "France is a country ... Paris is its capital city.  ││
│ │         <SEP> What is the capital of France?"               ││
│ │ ↓                                                            ││
│ │ Split at <SEP>:                                             ││
│ │   - Context: "France is a country ... Paris is its capital  ││
│ │              city."                                         ││
│ │   - Question: "What is the capital of France?"              ││
│ │ ↓                                                            ││
│ │ Process Context chunks → Extract K, V                       ││
│ │   - Chunk 1: [tokens 0-128] → K₁, V₁                       ││
│ │   - Chunk 2: [tokens 128-256] → K₂, V₂                     ││
│ │   - ... more chunks if needed                               ││
│ │ ↓                                                            ││
│ │ Stack all context K, V: [K₁; K₂; ...], [V₁; V₂; ...]       ││
│ │ ↓                                                            ││
│ │ SVD Compress to 128 tokens (YOUR CUSTOM)                    ││
│ │   - K_compressed (128 x hidden_dim)                         ││
│ │   - V_compressed (128 x hidden_dim)                         ││
│ │ ↓                                                            ││
│ │ Process Question chunks → Extract Q                         ││
│ │   - Question tokens → Q (question_length x hidden_dim)      ││
│ │ ↓                                                            ││
│ │ Final Cross-Attention (YOUR CUSTOM):                        ││
│ │   - Q (from question) attends to K_compressed, V_compressed ││
│ │ ↓                                                            ││
│ │ Output: encoder_K, encoder_V (question_length x hidden_dim) ││
│ └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘

                            ↓ (K, V passed to decoder)

┌─────────────────────────────────────────────────────────────────┐
│ Step 2: Decode                                                  │
│ ┌─────────────────────────────────────────────────────────────┐│
│ │ Input: "<start> Paris"                                      ││
│ │ ↓                                                            ││
│ │ GPT-OSS Decoder (if enabled)                                ││
│ │ ↓                                                            ││
│ │ Self-Attention (causal, only past tokens)                   ││
│ │ ↓                                                            ││
│ │ Cross-Attention to Encoder K,V (YOUR CUSTOM)                ││
│ │ ↓                                                            ││
│ │ Feed-Forward with Context (YOUR CUSTOM)                     ││
│ │ ↓                                                            ││
│ │ Output: "Paris is the capital"                              ││
│ └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Configuration Options

```
Option 1: No Pretrained Weights
┌────────────┐      ┌────────────┐
│  Encoder   │ ───▶ │  Decoder   │
│ (Trained)  │      │ (Trained)  │
└────────────┘      └────────────┘

Option 2: BERT Encoder Only
┌────────────┐      ┌────────────┐
│  Encoder   │ ───▶ │  Decoder   │
│   (BERT)   │      │ (Trained)  │
│  + Custom  │      │            │
└────────────┘      └────────────┘

Option 3: GPT-OSS Decoder Only
┌────────────┐      ┌────────────┐
│  Encoder   │ ───▶ │  Decoder   │
│ (Trained)  │      │  (GPT-OSS) │
│            │      │  + Custom  │
└────────────┘      └────────────┘

Option 4: Full Hybrid (BEST)
┌────────────┐      ┌────────────┐
│  Encoder   │ ───▶ │  Decoder   │
│   (BERT)   │      │  (GPT-OSS) │
│  + Custom  │      │  + Custom  │
└────────────┘      └────────────┘
```

## Custom Components (Always Preserved)

```
ENCODER CUSTOM:
  ✨ Context/Question Split  - Splits input at <SEP> token
  ✨ SVD Compression         - Compresses ALL context chunks' K,V to fixed length
  ✨ Final Cross-Attention   - Q (question) attends to compressed context K,V

DECODER CUSTOM:
  ✨ Context Projection     - Feeds previous output back
  ✨ Encoder-Decoder Cross-Attention - Attends to encoder
  ✨ Context State          - Maintains conversation state
```

## Memory Flow

```
Encoder:
  Input: "Context... <SEP> Question..."
       ↓
  Split at <SEP>:
    - Context tokens (variable: 512, 1024, 2048...)
    - Question tokens (variable: 10-50 typically)
       ↓
  Process CONTEXT chunks:
    Chunk 1 → K₁, V₁
    Chunk 2 → K₂, V₂
    ...
    Chunk N → Kₙ, Vₙ
       ↓
  Stack: [K₁; K₂; ...; Kₙ], [V₁; V₂; ...; Vₙ]
       ↓
  SVD Compress: ALL context K,V → 128 or 256 tokens ← YOUR CUSTOM ✨
       ↓
  Process QUESTION chunks:
    Extract Q from question tokens
       ↓
  Final Cross-Attention: Q (question) attends to compressed K,V (context)
       ↓
  Output: (question_length x hidden_dim) ← Sent to decoder

Decoder:
  Receives encoder output (question_length x hidden_dim)
       ↓
  Generates tokens autoregressively
       ↓
  Each new token attends to:
    - Previous tokens (self-attention)
    - Encoder output (cross-attention)
    - Previous context (YOUR CUSTOM)
```

This is the complete architecture! 🎯
