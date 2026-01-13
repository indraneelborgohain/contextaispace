# Fine-Tuning with GPT-OSS Weights

This guide shows you how to fine-tune your trained encoder-decoder model by optionally loading GPT-OSS weights for compatible layers.

## Step 1: Download GPT-OSS Weights

```bash
cd architecture/open-gpt-oss
python download_weights.py
cd ../..
```

**What this does:**
- Downloads ~20GB GPT-OSS model weights from Hugging Face
- Saves to `architecture/open-gpt-oss/weights/`
- Includes model weights, config, and tokenizer

**Requirements:**
```bash
pip install huggingface_hub
```

**Verify download:**
```bash
python check_models.py
```

---

## Step 2: Start Fine-Tuning

### Option A: Simple Fine-Tuning (Without GPT-OSS)

If you just want to fine-tune your existing model with new learning rates:

```bash
python run_pretrained_training.py
```

This will:
- ✅ Auto-find your latest checkpoint
- ✅ Use default learning rates (1e-5 for encoder and decoder)
- ✅ Ask for confirmation before starting

### Option B: Hybrid Loading (With GPT-OSS)

To load GPT-OSS weights for compatible layers while keeping your custom layers:

```bash
python train_with_pretrained.py \
    --checkpoint model_msmarco/checkpoint_5000.pt \
    --gptoss_weights architecture/open-gpt-oss/weights \
    --encoder_lr 1e-5 \
    --decoder_lr 1e-6 \
    --cross_attn_lr 3e-4 \
    --max_iters 5000
```

**What gets loaded from GPT-OSS:**
- ✅ Embedding layers
- ✅ Attention mechanisms  
- ✅ Base MLP layers

**What stays from your model:**
- ✅ Context projections
- ✅ Cross-attention layers
- ✅ Custom components (LSI, encoder adapters)

---

## Learning Rate Guide

| Approach | Encoder | Decoder | Custom Layers |
|----------|---------|---------|---------------|
| **Simple fine-tuning** | 1e-5 | 1e-5 | - |
| **With GPT-OSS (conservative)** | 1e-5 | 1e-6 | 3e-4 |
| **With GPT-OSS (aggressive)** | 3e-5 | 1e-5 | 5e-4 |

---

## Quick Commands

```bash
# 1. Download weights
cd architecture/open-gpt-oss && python download_weights.py && cd ../..

# 2. Check what you have
python check_models.py

# 3. Start fine-tuning (easy way)
python run_pretrained_training.py

# OR with GPT-OSS (advanced)
python train_with_pretrained.py \
    --checkpoint your_checkpoint.pt \
    --gptoss_weights architecture/open-gpt-oss/weights \
    --decoder_lr 1e-6
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
