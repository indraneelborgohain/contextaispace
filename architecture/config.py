"""
Model Configuration for Context-Aware GPT

This module contains the configuration dataclass for the GPT model.
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for context-aware GPT model."""
    num_hidden_layers: int = 24
    num_experts: int = 32
    experts_per_token: int = 4
    vocab_size: int = 201088
    hidden_size: int = 2880
    intermediate_size: int = 2880
    swiglu_limit: float = 7.0
    head_dim: int = 64
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    sliding_window: int = 128
    initial_context_length: int = 4096
    rope_theta: float = 150000.0
    rope_scaling_factor: float = 32.0
    rope_ntk_alpha: float = 1.0
    rope_ntk_beta: float = 32.0
    lsi_components: int = 64  # Number of LSI latent dimensions
    use_lsi_cross_attention: bool = True  # Enable LSI cross-attention
    use_context_embedding: bool = True  # Prepend context embedding to input sequence
    use_encoder_decoder_cross_attention: bool = False  # Enable encoder-decoder cross-attention
    encoder_hidden_size: int | None = None  # Hidden size of encoder (if different from decoder)
    
    @property
    def context_dim(self) -> int:
        return self.hidden_size
