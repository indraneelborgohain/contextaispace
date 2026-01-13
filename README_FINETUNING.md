# Fine-Tuning with Pretrained Weights (BERT + GPT-OSS)

This guide shows you how to fine-tune your encoder-decoder model by loading pretrained weights from open-source models:
- **BERT/RoBERTa** for the encoder
- **GPT-OSS** for the decoder

Your custom components (SVD compression, cross-attention) are preserved!

---

## Step 1: Download Pretrained Weights

### Option A: GPT-OSS Decoder Weights

```bash
cd architecture/open-gpt-oss
python download_weights.py
cd ../..
```

**What this downloads:**
- ~20GB GPT-OSS model weights from Hugging Face
- Saves to `architecture/open-gpt-oss/weights/`

### Option B: BERT Encoder Weights

BERT weights download automatically when you specify `--bert_model`:

```bash
# No manual download needed!
# Will auto-download when you run training with --bert_model
```

**Common BERT models:**
- `bert-base-uncased` (110M params)
- `bert-large-uncased` (340M params)
- `roberta-base` (125M params)
- `roberta-large` (355M params)

**Requirements:**
```bash
pip install transformers huggingface_hub
```

**Verify download:**
```bash
python check_models.py
```

---

## Step 2: Start Fine-Tuning

### Option A: Simple Fine-Tuning (No Pretrained Weights)

Just fine-tune your existing model with new learning rates:

```bash
python run_pretrained_training.py
```

### Option B: Hybrid Decoder (GPT-OSS)

Load GPT-OSS weights for decoder base layers:

```bash
python train_with_pretrained.py \
    --checkpoint model_msmarco/checkpoint_5000.pt \
    --gptoss_weights architecture/open-gpt-oss/weights \
    --encoder_lr 1e-5 \
    --decoder_lr 1e-6 \
    --cross_attn_lr 3e-4
```

**Decoder layers loaded from GPT-OSS:**
- ✅ Embedding, attention, base MLP

**Decoder layers kept from your model:**
- ✅ Context projections, cross-attention, custom components

### Option C: Hybrid Encoder (BERT)

Load BERT weights for encoder base layers:

```bash
python train_with_pretrained.py \
    --checkpoint model_msmarco/checkpoint_5000.pt \
    --bert_model bert-base-uncased \
    --encoder_lr 1e-5 \
    --decoder_lr 1e-5
```

**Encoder layers loaded from BERT:**
- ✅ Embedding, attention, FFN

**Encoder layers kept from your model:**
- ✅ SVD compression, cross-attention between chunks

### Option D: Full Hybrid (BERT + GPT-OSS) 🚀

Load both BERT for encoder AND GPT-OSS for decoder:

```bash
python train_with_pretrained.py \
    --checkpoint model_msmarco/checkpoint_5000.pt \
    --bert_model bert-base-uncased \
    --gptoss_weights architecture/open-gpt-oss/weights \
    --encoder_lr 1e-5 \
    --decoder_lr 1e-6 \
    --cross_attn_lr 3e-4 \
    --max_iters 5000
```

**This combines:**
- 🤗 BERT's language understanding (encoder)
- 🤖 GPT-OSS's generation power (decoder)  
- 🎯 Your custom architecture (SVD compression + cross-attention)

---

## Learning Rate Guide

| Approach | Encoder | Decoder | Custom Layers | Use Case |
|----------|---------|---------|---------------|----------|
| **Simple** | 1e-5 | 1e-5 | - | Continue training |
| **BERT only** | 1e-6 | 1e-5 | 3e-4 | Improve encoding |
| **GPT-OSS only** | 1e-5 | 1e-6 | 3e-4 | Improve generation |
| **Both (recommended)** | 1e-6 | 1e-6 | 3e-4 | Best of both worlds |

---

## Quick Commands

```bash
# 1. Download GPT-OSS (optional)
cd architecture/open-gpt-oss && python download_weights.py && cd ../..

# 2. Check what you have
python check_models.py

# 3a. Simple fine-tuning
python run_pretrained_training.py

# 3b. With BERT encoder
python train_with_pretrained.py \
    --checkpoint your_checkpoint.pt \
    --bert_model bert-base-uncased \
    --encoder_lr 1e-6

# 3c. With GPT-OSS decoder  
python train_with_pretrained.py \
    --checkpoint your_checkpoint.pt \
    --gptoss_weights architecture/open-gpt-oss/weights \
    --decoder_lr 1e-6

# 3d. With BOTH (full hybrid) 🚀
python train_with_pretrained.py \
    --checkpoint your_checkpoint.pt \
    --bert_model roberta-base \
    --gptoss_weights architecture/open-gpt-oss/weights \
    --encoder_lr 1e-6 \
    --decoder_lr 1e-6 \
    --cross_attn_lr 3e-4
```

---

## Monitoring

```bash
# View logs in terminal
# Loss and learning rates printed every 10 iterations

# Or use TensorBoard
tensorboard --logdir runs_pretrained
```

---

## Output

Fine-tuned models saved to `model_finetuned/`:
- `checkpoint_500.pt`
- `checkpoint_1000.pt`
- `final_model.pt`

Each contains encoder, decoder, optimizer state, and config.

---

## Troubleshooting

**"No checkpoint found"**
```bash
# Train a model first
python train_msmarco.py --max_iters 1000
```

**"GPT-OSS weights not found"**
```bash
# Repeat Step 1
cd architecture/open-gpt-oss
python download_weights.py
```

**"CUDA out of memory"**
```bash
# Reduce batch size
python train_with_pretrained.py --checkpoint your.pt --batch_size 2
```

**Download is slow**
- GPT-OSS is ~20GB, download time depends on your internet speed
- Be patient or use a faster connection

---

## Need Help?

- See [QUICKSTART.md](QUICKSTART.md) for detailed guide
- See [ARCHITECTURE_COMPATIBILITY.md](ARCHITECTURE_COMPATIBILITY.md) for technical details
- Run `python check_models.py` to see what checkpoints you have
