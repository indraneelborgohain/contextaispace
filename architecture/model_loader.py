"""Helper functions for loading encoder-decoder models."""

from .decoder import Transformer, ModelConfig


def load_decoder(vocab_size, device, encoder_hidden_size=None):
    """
    Create decoder with optional cross-attention to encoder.
    Uses default ModelConfig values from the ModelConfig class.
    
    Args:
        vocab_size: Vocabulary size for the decoder
        device: torch device
        encoder_hidden_size: If provided, enables cross-attention to encoder (e.g., 1024 for BERT-large)
    
    Returns:
        Transformer: Decoder model
    """
    print(f"\n{'='*60}")
    print("Creating Decoder")
    print(f"{'='*60}")
    if encoder_hidden_size:
        print(f"Cross-attention enabled (encoder hidden size: {encoder_hidden_size})")
    
    # Create decoder config using defaults from ModelConfig
    decoder_config = ModelConfig(vocab_size=vocab_size)
    
    print(f"\nCreating decoder with default config...")
    print(f"  Layers: {decoder_config.num_hidden_layers}")
    print(f"  Hidden size: {decoder_config.hidden_size}")
    print(f"  Intermediate size: {decoder_config.intermediate_size}")
    print(f"  Num experts: {decoder_config.num_experts}")
    print(f"  Experts per token: {decoder_config.experts_per_token}")
    print(f"  Attention heads: {decoder_config.num_attention_heads}")
    print(f"  KV heads: {decoder_config.num_key_value_heads}")
    
    decoder = Transformer(decoder_config, encoder_hidden_size=encoder_hidden_size, device=device)
    
    print(f"\nDecoder parameters: {sum(p.numel() for p in decoder.parameters())/1e6:.1f}M\n")
    
    return decoder
