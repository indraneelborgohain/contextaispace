# Training with Pretrained Weights

This guide shows how to use `train_with_pretrained.py` to retrain a model using:
- **GPT-OSS weights** for the decoder
- **Your trained encoder** weights
- **New learning rates** for fine-tuning

## Quick Start

### 1. Prerequisites

Make sure you have:
- Trained encoder checkpoint (`.pt` file)
- GPT-OSS weights downloaded (run if needed):
  ```bash
  python architecture/open-gpt-oss/download_weights.py
  ```

### 2. Basic Usage

```bash
python train_with_pretrained.py \
    --encoder_checkpoint path/to/your/encoder_checkpoint.pt \
    --gptoss_weights architecture/open-gpt-oss/weights \
    --out_dir model_pretrained \
    --encoder_lr 1e-5 \
    --decoder_lr 1e-6 \
    --cross_attn_lr 3e-4 \
    --batch_size 4 \
    --max_iters 5000
```

### 3. Learning Rate Strategy

The script uses **differential learning rates** for different components:

- **Encoder LR** (`--encoder_lr 1e-5`): Low rate for your trained encoder
- **Decoder LR** (`--decoder_lr 1e-6`): Very low rate for pretrained GPT-OSS (frozen-ish)
- **Cross-Attention LR** (`--cross_attn_lr 3e-4`): Normal rate for new cross-attention layers

This prevents catastrophic forgetting of pretrained weights while allowing new components to learn.

### 4. Full Options

```bash
python train_with_pretrained.py \
    # Model paths
    --encoder_checkpoint path/to/encoder.pt \
    --gptoss_weights architecture/open-gpt-oss/weights \
    --out_dir model_pretrained \
    
    # Learning rates (key parameters!)
    --encoder_lr 1e-5 \
    --decoder_lr 1e-6 \
    --cross_attn_lr 3e-4 \
    
    # Training
    --batch_size 4 \
    --max_iters 5000 \
    --max_context_len 512 \
    --max_qa_len 128 \
    
    # Optimizer
    --weight_decay 0.1 \
    --grad_clip 1.0 \
    --warmup_iters 200 \
    --min_lr_ratio 0.1 \
    
    # System
    --device cuda:0 \
    --dtype bfloat16 \
    --seed 123 \
    
    # Dataset
    --dataset msmarco \
    
    # Logging
    --use_tensorboard \
    --log_dir runs_pretrained \
    --log_interval 10 \
    --save_every 500
```

### 5. What the Script Does

1. **Loads your trained encoder** from checkpoint
2. **Loads GPT-OSS decoder** weights from the weights directory
3. **Wraps GPT-OSS with cross-attention** layers to connect with encoder
4. **Creates optimizer** with 3 different learning rate groups:
   - Pretrained encoder (low LR)
   - Pretrained GPT-OSS decoder (very low LR)
   - New cross-attention layers (normal LR)
5. **Trains** with cosine LR scheduling and warmup

### 6. Finding Your Encoder Checkpoint

To find your trained encoder checkpoints:

```powershell
# Find all .pt files
Get-ChildItem -Path "c:\git\contextaispace" -Recurse -Filter "*.pt"

# Find checkpoints with details
Get-ChildItem -Path "c:\git\contextaispace" -Recurse -Filter "checkpoint*.pt" | 
    Select-Object FullName, @{Name="SizeMB";Expression={[math]::Round($_.Length/1MB, 2)}}, LastWriteTime
```

### 7. Monitoring Training

With TensorBoard:
```bash
tensorboard --logdir runs_pretrained
```

The script logs:
- Training loss
- Individual learning rates for each component
- Validation metrics (if implemented)

### 8. Next Steps

After running the script, you'll need to implement:
- `get_training_batch()` function for your specific data format
- Forward pass logic with encoder → decoder cross-attention
- Validation loop (optional)

The script provides the framework for loading pretrained weights and managing differential learning rates.

## Example: Training on MS MARCO

```bash
# Assuming you have:
# - Encoder checkpoint: model_msmarco/checkpoint_5000.pt
# - GPT-OSS weights: architecture/open-gpt-oss/weights/

python train_with_pretrained.py \
    --encoder_checkpoint model_msmarco/checkpoint_5000.pt \
    --gptoss_weights architecture/open-gpt-oss/weights \
    --out_dir model_gptoss_finetuned \
    --encoder_lr 5e-6 \
    --decoder_lr 5e-7 \
    --cross_attn_lr 1e-4 \
    --batch_size 2 \
    --max_iters 10000 \
    --use_tensorboard
```

## Tips for Fine-Tuning

1. **Start with very low decoder LR** (1e-6 or lower) to preserve GPT-OSS knowledge
2. **Higher cross-attention LR** (1e-4 to 3e-4) since these are new layers
3. **Gradual warmup** helps prevent early instability
4. **Monitor loss carefully** - if it spikes, lower the decoder LR
5. **Use gradient clipping** (1.0) to prevent explosions
