from training.data_loader import train_loader, val_loader
from architecture.gptoss import ModelConfig
from architecture.model_loader import (
    load_pretrained_gptoss, 
    list_available_checkpoints,
    get_model_info
)
import torch
from inference import generate_text

device = "cuda:0"
context = "Once upon a day"

# Configuration for the model
config = ModelConfig(
    num_attention_heads=8,
    num_key_value_heads=4,
    num_experts=4,
    experts_per_token=1,
    num_hidden_layers=12,
    hidden_size=1024,
    intermediate_size=1024
)

# List available checkpoints
list_available_checkpoints(models_dir="models")

# Load pretrained model (or initialize with random weights if no checkpoint exists)
model = load_pretrained_gptoss(
    checkpoint_path="models/gptoss_best.pt",
    config=config,
    device=device,
    strict=False  # Set to True if you want exact state_dict matching
)

# Display model information
get_model_info(model)
generate_text(model,context)
# Start training
from training.trainer import trainer
tl, vl, ts = trainer(model, train_loader, val_loader, device)
