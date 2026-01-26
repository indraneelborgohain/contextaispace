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
     num_hidden_layers = 24
    num_experts = 32
    experts_per_token  = 4
    vocab_size = 201088
    hidden_size  = 2880
    intermediate_size  = 2880
    swiglu_limit  = 7.0
    head_dim = 64
    num_attention_heads = 64
    num_key_value_heads = 8
    sliding_window = 128
    initial_context_length = 4096
    rope_theta = 150000.0
    rope_scaling_factor = 32.0
    rope_ntk_alpha = 1.0
    rope_ntk_beta = 32.0
)

# List available checkpoints
list_available_checkpoints(models_dir="models")

# Load pretrained model (or initialize with random weights if no checkpoint exists)
model = load_pretrained_gptoss(
    checkpoint_path="models/model.safetensors",
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
