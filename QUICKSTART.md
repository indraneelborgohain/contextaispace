# Quick Start: Fine-Tuning with Optional GPT-OSS Hybrid Loading

## Two Approaches Available

### Approach 1: Pure Fine-Tuning (Simple)
Load your trained checkpoint and fine-tune with new learning rates.

### Approach 2: Hybrid GPT-OSS Loading (Advanced)
Load GPT-OSS weights for compatible layers (embedding, attention, MLP) and keep your trained weights for custom layers (context projections, cross-attention).

## Hybrid Loading Explained

Your decoder has a custom architecture with unique features. The hybrid approach:
- ✅ **Loads from GPT-OSS**: Embedding, attention layers, standard MLP components
- ✅ **Keeps your trained weights**: Context projections, cross-attention, LSI, custom layers
- ✅ **Best of both worlds**: GPT-OSS knowledge + your custom functionality

### What Gets Loaded from GPT-OSS:
- `embedding.weight`
- `block.*.attn.*` (where shapes match)
- `block.*.mlp.*` (base MLP layers)
- `norm.*`
- `unembedding.*`

### What Stays from Your Model:
- `*.context_proj.*` (context-aware mechanism)
- `*.cross_attn.*` (encoder-decoder attention)
- `start_token_embedding` (your custom start token)
- `*.lsi.*` (LSI compression if enabled)
- `*.encoder_projection.*` (encoder adapters)

## Step-by-Step Guide

### Step 1: Check Your Models
```bash
python check_models.py
```

This will show you available trained checkpoints.

### Step 2: Run Fine-Tuning

#### Option A: Simple Fine-Tuning (No GPT-OSS)
```bash
python run_pretrained_training.py
```

#### Option B: Hybrid with GPT-OSS Weights
First download GPT-OSS weights:
```bash
cd architecture/open-gpt-oss
python download_weights.py
cd ../..
```

Then run with hybrid loading:
```bash
python train_with_pretrained.py \
    --checkpoint path/to/your/checkpoint.pt \
    --gptoss_weights architecture/open-gpt-oss/weights \
    --encoder_lr 1e-5 \
    --decoder_lr 1e-6 \
    --cross_attn_lr 3e-4 \
    --batch_size 4 \
    --max_iters 5000
#### For Pure Fine-Tuning (2-tier):
| Component | Learning Rate | Why? |
|-----------|---------------|------|
| **Encoder** | 1e-5 | Already trained, gentle adjustments |
| **Decoder** | 1e-5 | Already trained, gentle adjustments |

#### For Hybrid GPT-OSS (3-tier):
| Component | Learning Rate | Why? |
|-----------|---------------|------|
| **Encoder** | 1e-5 | Your trained weights, gentle |
| **Decoder Base** | 1e-6 | GPT-OSS weights, preserve knowledge |
| **Custom Layers** | 3e-4 | Your layers, can adapt more
- Use 3-tier learning rates

### Step 3: Understanding the Learning Rates

For fine-tuning, use **lower learning rates** than initial training:

| Component | Typical Initial | Fine-Tuning | Why? |
|-----------|----------------|-------------|------|
| **Encoder** | 3e-4 | 1e-5 | Already trained, gentle adjustments |
| **Decoder** | 3e-4 | 1e-5 | Already trained, gentle adjustments |

### Step 4: What If I Don't Have a Trained Model?

First train a model using one of these:

```bash
# Train on MS MARCO (recommended for Q&A)
python train_msmarco.py --max_iters 1000 --out_dir model_msmarco

# OR train on SQuAD
python train_squad.py --max_iters 1000 --out_dir model_squad

# OR train on stories
python train_encoder_decoder_stories.py --max_iters 1000
```

Then proceed to Step 2!

## Advanced: Custom Learning Rates

### Conservative (minimal changes)
```bash
--encoder_lr 5e-6 \
--decoder_lr 5e-6
```

### Moderate (default)
```bash
--encoder_lr 1e-5 \
--decoder_lr 1e-5
```

### Aggressive (more adaptation)
```bash
--encoder_lr 3e-5 \
--decoder_lr 3e-5
```

### Freeze Encoder (only train decoder)
```bash
--encoder_lr 0 \
--decoder_lr 1e-5
```

## Use Cases for Fine-Tuning

1. **Continued Training**: Train for more iterations with lower LR
2. **Domain Adaptation**: Adapt to a specific domain/dataset
3. **Learning Rate Adjustment**: Resume with better LR if training diverged
4. **Task Transfer**: Fine-tune from one task to another

## Monitoring Training

### View Real-time Loss
Watch the terminal for:
```
iter  1000 | loss 2.3456 | enc_lr 1.00e-05 | dec_lr 1.00e-05 | 45.23ms
```

### Use TensorBoard
```bash
# In another terminal
tensorboard --logdir runs_pretrained

# Open: http://localhost:6006
```

## Troubleshooting

### "No checkpoint files found"
Train a model first (see Step 4)

### "CUDA out of memory"
Reduce batch size:
```bash
--batch_size 2  # or even 1
```

### Loss is not improving
Try different learning rates:
```bash
# Lower
--encoder_lr 5e-6 --decoder_lr 5e-6

# Or higher
--encoder_lr 3e-5 --decoder_lr 3e-5
```

## What Happens Next?

After fine-tuning completes, you'll have:
- `model_finetuned/checkpoint_500.pt`
- `model_finetuned/checkpoint_1000.pt`
- ...
- `model_finetuned/final_model.pt`

Each checkpoint contains:
- ✅ Encoder weights (fine-tuned)
- ✅ Decoder weights (fine-tuned)
- ✅ Optimizer state
- ✅ Training metadata

## Files Available

1. **[train_with_pretrained.py](train_with_pretrained.py)** - Main fine-tuning script
2. **[run_pretrained_training.py](run_pretrained_training.py)** - Easy launcher
3. **[check_models.py](check_models.py)** - Check available checkpoints
4. **QUICKSTART.md** - This file!

## Ready to Start?

```bash
# 1. Check what you have
python check_models.py

# 2. Start fine-tuning!
python run_pretrained_training.py
```

Good luck! 🚀
