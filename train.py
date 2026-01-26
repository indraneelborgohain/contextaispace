
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
config = ModelConfig()

# List available checkpoints
list_available_checkpoints(models_dir="models")

# Load pretrained model (or initialize with random weights if no checkpoint exists)
model = load_pretrained_gptoss(
    checkpoint_path="models/gptoss_best.safetensors",
    config=config,
    device=device,
    strict=False  # Set to True if you want exact state_dict matching
)

# Display model information
get_model_info(model)

generate_text(model, context)
# Start training
from training.trainer import trainer
from training.data_loader import train_loader, val_loader
tl, vl, ts = trainer(model, train_loader, val_loader, device)
