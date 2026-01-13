# BERT Encoder Hybrid Loading Summary

## What's New

You can now load **BERT/RoBERTa weights** for your encoder while keeping your custom SVD compression and cross-attention logic!

## How It Works

### Encoder Architecture Mapping

**From BERT (loaded):**
- ✅ `embeddings.word_embeddings` → Your `embedding`
- ✅ `encoder.layer.N.attention` → Your `blocks.N.attn`
- ✅ `encoder.layer.N.intermediate` → Your `blocks.N.mlp`
- ✅ Layer normalizations

**From Your Trained Model (preserved):**
- ✅ `cross_attn` - Cross-attention between chunks
- ✅ `final_cross_attn` - Final chunk aggregation
- ✅ `_compress_with_svd` - SVD compression logic
- ✅ LSI compression (if enabled)

## Usage

### 1. BERT Encoder Only

```bash
python train_with_pretrained.py \
    --checkpoint your_checkpoint.pt \
    --bert_model bert-base-uncased \
    --encoder_lr 1e-6 \
    --decoder_lr 1e-5 \
    --cross_attn_lr 3e-4
```

### 2. GPT-OSS Decoder Only

```bash
python train_with_pretrained.py \
    --checkpoint your_checkpoint.pt \
    --gptoss_weights architecture/open-gpt-oss/weights \
    --encoder_lr 1e-5 \
    --decoder_lr 1e-6 \
    --cross_attn_lr 3e-4
```

### 3. Full Hybrid (BERT + GPT-OSS) 🚀

```bash
python train_with_pretrained.py \
    --checkpoint your_checkpoint.pt \
    --bert_model roberta-base \
    --gptoss_weights architecture/open-gpt-oss/weights \
    --encoder_lr 1e-6 \
    --decoder_lr 1e-6 \
    --cross_attn_lr 3e-4
```

## Popular BERT Models

| Model | Size | Use Case |
|-------|------|----------|
| `bert-base-uncased` | 110M | General purpose, fast |
| `bert-large-uncased` | 340M | Better quality, slower |
| `roberta-base` | 125M | Better than BERT base |
| `roberta-large` | 355M | Best quality |
| `distilbert-base-uncased` | 66M | Fastest, good quality |

## Why This is Powerful

### Before (Your Original Model)
- Encoder: Trained from scratch on your data
- Decoder: Trained from scratch on your data
- Custom: SVD compression + cross-attention

### After (Full Hybrid)
- Encoder: **BERT's pretrained language understanding**
- Decoder: **GPT-OSS's pretrained generation**
- Custom: **Your SVD compression + cross-attention**

### Result
🎯 **Best of three worlds combined!**

## Learning Rate Strategy

| Component | LR | Reasoning |
|-----------|-----|-----------|
| BERT encoder | 1e-6 | Very low - preserve pretrained knowledge |
| GPT-OSS decoder | 1e-6 | Very low - preserve pretrained knowledge |
| Your custom layers | 3e-4 | Normal - learn to connect components |

## What Gets Preserved

Your custom architecture features are NOT overwritten:

1. **SVD Compression** - Your `_compress_with_svd()` method stays intact
2. **Cross-Attention** - Your `cross_attn` and `final_cross_attn` layers
3. **Chunk Processing** - Your reverse-order chunk logic
4. **LSI Compression** - If you're using LSI instead of SVD

## Installation

```bash
pip install transformers huggingface_hub
```

BERT weights download automatically when you specify `--bert_model`.

## Complete Example

```bash
# 1. Check your setup
python check_models.py
python example_full_hybrid.py

# 2. Download GPT-OSS (if using)
cd architecture/open-gpt-oss
python download_weights.py
cd ../..

# 3. Run full hybrid training
python train_with_pretrained.py \
    --checkpoint model_msmarco/checkpoint_5000.pt \
    --bert_model roberta-base \
    --gptoss_weights architecture/open-gpt-oss/weights \
    --encoder_lr 1e-6 \
    --decoder_lr 1e-6 \
    --cross_attn_lr 3e-4 \
    --batch_size 4 \
    --max_iters 5000 \
    --use_tensorboard
```

## Files Updated

- ✅ [train_with_pretrained.py](train_with_pretrained.py) - Added `load_bert_weights_partial()` and `--bert_model` arg
- ✅ [README_FINETUNING.md](README_FINETUNING.md) - Updated with BERT options
- ✅ [example_full_hybrid.py](example_full_hybrid.py) - Interactive example
- ✅ **BERT_HYBRID.md** - This file!

## Summary

You can now choose:
- ✅ Simple fine-tuning (no pretrained)
- ✅ BERT encoder hybrid
- ✅ GPT-OSS decoder hybrid
- ✅ **Full hybrid (BERT + GPT-OSS)** ← Most powerful!

All while keeping your custom SVD compression and cross-attention intact! 🎉
