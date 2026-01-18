"""Context-Aware GPT Implementation

This module implements a nano-GPT OSS variant where the output context vector 
from the previous prediction is fed back as input to the model.

The context state (shape: hidden_size) stores the output from the previous token 
prediction and is fed into each transformer layer through context projection layers.
"""

import json
import math
import os

import torch
import torch.distributed as dist
import numpy as np
from .config import ModelConfig
from .lsi import LatentSemanticIndexing, LSIConfig
from .attention_components import RMSNorm, RotaryEmbedding, ContextAttentionBlock, sdpa


def swiglu(x, alpha: float = 1.702, limit: float = 7.0):
    """SwiGLU activation with clamping."""
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    return out_glu * (x_linear + 1)


class ContextMLPBlock(torch.nn.Module):
    """
    Context-aware Mixture of Experts MLP block.
    Uses sparse MoE with top-k expert selection.
    """
    def __init__(
        self,
        config: ModelConfig,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.num_experts = config.num_experts
        self.experts_per_token = config.experts_per_token
        self.swiglu_limit = config.swiglu_limit
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.norm = RMSNorm(config.hidden_size, device=device)
        
        self.context_proj = torch.nn.Linear(
            config.context_dim, config.hidden_size, device=device, dtype=torch.bfloat16
        )
        
        self.gate = torch.nn.Linear(
            config.hidden_size, config.num_experts, device=device, dtype=torch.bfloat16
        )
        assert config.intermediate_size % self.world_size == 0
        
        self.experts = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(
                    config.hidden_size, 
                    config.intermediate_size * 2 // self.world_size, 
                    device=device, 
                    dtype=torch.bfloat16
                ),
                torch.nn.Linear(
                    config.intermediate_size // self.world_size, 
                    config.hidden_size, 
                    device=device, 
                    dtype=torch.bfloat16
                )
            ) for _ in range(config.num_experts)
        ])

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        seq_len, hidden_size = x.shape
        
        context_proj = self.context_proj(context)
        x_with_context = x + context_proj
        
        t = self.norm(x_with_context)
        g = self.gate(t)
        
        experts = torch.topk(g, k=self.experts_per_token, dim=-1, sorted=True)
        expert_weights = torch.nn.functional.softmax(experts.values, dim=-1)
        expert_indices = experts.indices
        
        t_flat = t.view(-1, hidden_size)
        expert_indices_flat = expert_indices.view(-1, self.experts_per_token)
        expert_weights_flat = expert_weights.view(-1, self.experts_per_token)
        
        output = torch.zeros_like(t_flat)
        
        for expert_idx in range(self.num_experts):
            mask = (expert_indices_flat == expert_idx).any(dim=-1)
            if not mask.any():
                continue
                
            token_indices = torch.where(mask)[0]
            expert_pos = (expert_indices_flat[token_indices] == expert_idx).nonzero(as_tuple=True)[1]
            
            expert_input = t_flat[token_indices]
            weights = expert_weights_flat[token_indices, expert_pos]
            
            expert_out = expert_input
            expert_out = self.experts[expert_idx][0](expert_out)
            expert_out = swiglu(expert_out, limit=self.swiglu_limit)
            expert_out = self.experts[expert_idx][1](expert_out)
            
            output[token_indices] += expert_out * weights.unsqueeze(-1)
        
        if self.world_size > 1:
            dist.all_reduce(output, op=dist.ReduceOp.SUM)
        
        output = output.view(seq_len, hidden_size)
        return x + output


class ContextTransformerBlock(torch.nn.Module):
    """Transformer block with context-aware attention and MLP."""
    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = ContextAttentionBlock(config, layer_idx, device)
        self.mlp = ContextMLPBlock(config, device)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = self.attn(x, context)
        x = self.mlp(x, context)
        return x


class AttentionBlock(torch.nn.Module):
    """
    Standard attention block without context.
    Grouped-query attention with RoPE.
    """
    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int = 0,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.sliding_window = config.sliding_window if layer_idx % 2 == 0 else 0
        self.sinks = torch.nn.Parameter(
            torch.empty(config.num_attention_heads, device=device, dtype=torch.bfloat16)
        )
        self.norm = RMSNorm(config.hidden_size, device=device)
        
        qkv_dim = config.head_dim * (
            config.num_attention_heads + 2 * config.num_key_value_heads
        )
        self.qkv = torch.nn.Linear(
            config.hidden_size, qkv_dim, device=device, dtype=torch.bfloat16
        )
        self.out = torch.nn.Linear(
            config.head_dim * config.num_attention_heads,
            config.hidden_size,
            device=device,
            dtype=torch.bfloat16,
        )
        self.sm_scale = 1 / math.sqrt(config.head_dim)
        self.rope = RotaryEmbedding(
            config.head_dim,
            config.rope_theta,
            torch.float32,
            initial_context_length=config.initial_context_length,
            scaling_factor=config.rope_scaling_factor,
            ntk_alpha=config.rope_ntk_alpha,
            ntk_beta=config.rope_ntk_beta,
            device=device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.norm(x)
        qkv = self.qkv(t)
        q = qkv[:, : self.num_attention_heads * self.head_dim].contiguous()
        k = qkv[
            :,
            self.num_attention_heads
            * self.head_dim : (self.num_attention_heads + self.num_key_value_heads)
            * self.head_dim,
        ].contiguous()
        v = qkv[
            :,
            (self.num_attention_heads + self.num_key_value_heads)
            * self.head_dim : (self.num_attention_heads + 2 * self.num_key_value_heads)
            * self.head_dim,
        ].contiguous()

        q = q.view(
            -1,
            self.num_key_value_heads,
            self.num_attention_heads // self.num_key_value_heads,
            self.head_dim,
        )
        k = k.view(-1, self.num_key_value_heads, self.head_dim)
        v = v.view(-1, self.num_key_value_heads, self.head_dim)
        q, k = self.rope(q, k)
        t = sdpa(q, k, v, self.sinks, self.sm_scale, self.sliding_window)
        t = self.out(t)
        t = x + t
        return t


class MLPBlock(torch.nn.Module):
    """
    Standard Mixture of Experts MLP block without context.
    Uses sparse MoE with top-k expert selection.
    """
    def __init__(
        self,
        config: ModelConfig,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.num_experts = config.num_experts
        self.experts_per_token = config.experts_per_token
        self.swiglu_limit = config.swiglu_limit
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.norm = RMSNorm(config.hidden_size, device=device)
        
        self.gate = torch.nn.Linear(
            config.hidden_size, config.num_experts, device=device, dtype=torch.bfloat16
        )
        assert config.intermediate_size % self.world_size == 0
        
        self.experts = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(
                    config.hidden_size, 
                    config.intermediate_size * 2 // self.world_size, 
                    device=device, 
                    dtype=torch.bfloat16
                ),
                torch.nn.Linear(
                    config.intermediate_size // self.world_size, 
                    config.hidden_size, 
                    device=device, 
                    dtype=torch.bfloat16
                )
            ) for _ in range(config.num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len, hidden_size = x.shape
        
        t = self.norm(x)
        g = self.gate(t)
        
        experts = torch.topk(g, k=self.experts_per_token, dim=-1, sorted=True)
        expert_weights = torch.nn.functional.softmax(experts.values, dim=-1)
        expert_indices = experts.indices
        
        t_flat = t.view(-1, hidden_size)
        expert_indices_flat = expert_indices.view(-1, self.experts_per_token)
        expert_weights_flat = expert_weights.view(-1, self.experts_per_token)
        
        output = torch.zeros_like(t_flat)
        
        for expert_idx in range(self.num_experts):
            mask = (expert_indices_flat == expert_idx).any(dim=-1)
            if not mask.any():
                continue
                
            token_indices = torch.where(mask)[0]
            expert_pos = (expert_indices_flat[token_indices] == expert_idx).nonzero(as_tuple=True)[1]
            
            expert_input = t_flat[token_indices]
            weights = expert_weights_flat[token_indices, expert_pos]
            
            expert_out = expert_input
            expert_out = self.experts[expert_idx][0](expert_out)
            expert_out = swiglu(expert_out, limit=self.swiglu_limit)
            expert_out = self.experts[expert_idx][1](expert_out)
            
            output[token_indices] += expert_out * weights.unsqueeze(-1)
        
        if self.world_size > 1:
            dist.all_reduce(output, op=dist.ReduceOp.SUM)
        
        output = output.view(seq_len, hidden_size)
        return x + output


class TransformerBlock(torch.nn.Module):
    """Standard transformer block without context, but with optional encoder cross-attention."""
    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = AttentionBlock(config, layer_idx, device)
        self.mlp = MLPBlock(config, device)
        
        # Optional cross-attention to encoder
        if config.use_encoder_decoder_cross_attention:
            encoder_size = config.encoder_hidden_size if config.encoder_hidden_size else config.hidden_size
            self.cross_attn = CrossAttentionLayer(config, encoder_hidden_size=encoder_size, device=device)
        else:
            self.cross_attn = None

    def forward(
        self, 
        x: torch.Tensor,
        encoder_k: torch.Tensor | None = None,
        encoder_v: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Self-attention
        x = self.attn(x)
        
        # Cross-attention (if enabled and encoder K, V provided)
        if self.cross_attn is not None and encoder_k is not None and encoder_v is not None:
            x = self.cross_attn(x, encoder_k, encoder_v)
        
        # MLP
        x = self.mlp(x)
        return x


class CrossAttentionLayer(torch.nn.Module):
    """
    Cross-attention layer for decoder to attend to encoder outputs.
    
    Q: from decoder hidden states
    K, V: from encoder (compressed via SVD)
    """
    def __init__(
        self,
        config: ModelConfig,
        encoder_hidden_size: int | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        
        # If encoder has different hidden size, use that for K,V projections
        if encoder_hidden_size is None:
            encoder_hidden_size = config.hidden_size
        
        self.norm = RMSNorm(config.hidden_size, device=device)
        
        # Q projection (from decoder)
        self.q_proj = torch.nn.Linear(
            config.hidden_size,
            config.head_dim * config.num_attention_heads,
            device=device,
            dtype=torch.bfloat16
        )
        
        # K, V projections (from encoder) - use encoder hidden size
        self.k_proj = torch.nn.Linear(
            encoder_hidden_size,
            config.head_dim * config.num_key_value_heads,
            device=device,
            dtype=torch.bfloat16
        )
        self.v_proj = torch.nn.Linear(
            encoder_hidden_size,
            config.head_dim * config.num_key_value_heads,
            device=device,
            dtype=torch.bfloat16
        )
        
        # Output projection
        self.out = torch.nn.Linear(
            config.head_dim * config.num_attention_heads,
            config.hidden_size,
            device=device,
            dtype=torch.bfloat16,
        )
        
        self.sm_scale = 1 / math.sqrt(config.head_dim)
    
    def forward(
        self, 
        x: torch.Tensor, 
        encoder_k: torch.Tensor, 
        encoder_v: torch.Tensor
    ) -> torch.Tensor:
        """
        Cross-attention forward pass.
        
        Args:
            x: Decoder hidden states (seq_len, hidden_size)
            encoder_k: Encoder keys (enc_seq_len, hidden_size) - SVD compressed
            encoder_v: Encoder values (enc_seq_len, hidden_size) - SVD compressed
        
        Returns:
            Output (seq_len, hidden_size)
        """
        residual = x
        x = self.norm(x)
        
        seq_len = x.shape[0]
        enc_seq_len = encoder_k.shape[0]
        
        # Project queries from decoder
        q = self.q_proj(x)  # (seq_len, num_heads * head_dim)
        
        # Project keys and values from encoder
        k = self.k_proj(encoder_k)  # (enc_seq_len, num_kv_heads * head_dim)
        v = self.v_proj(encoder_v)  # (enc_seq_len, num_kv_heads * head_dim)
        
        # Reshape for multi-head attention
        q = q.view(
            seq_len,
            self.num_key_value_heads,
            self.num_attention_heads // self.num_key_value_heads,
            self.head_dim
        )  # (seq_len, num_kv_heads, q_mult, head_dim)
        
        k = k.view(enc_seq_len, self.num_key_value_heads, self.head_dim)
        v = v.view(enc_seq_len, self.num_key_value_heads, self.head_dim)
        
        # Grouped query attention
        # Expand K and V to match Q's group dimension
        q_mult = self.num_attention_heads // self.num_key_value_heads
        k = k[:, :, None, :].expand(-1, -1, q_mult, -1)  # (enc_seq_len, num_kv_heads, q_mult, head_dim)
        v = v[:, :, None, :].expand(-1, -1, q_mult, -1)
        
        # Compute attention scores: Q @ K^T
        # q: (seq_len, num_kv_heads, q_mult, head_dim)
        # k: (enc_seq_len, num_kv_heads, q_mult, head_dim)
        attn_scores = torch.einsum('qhmd,khmd->hmqk', q, k)  # (num_kv_heads, q_mult, seq_len, enc_seq_len)
        attn_scores = attn_scores * self.sm_scale
        
        # No causal mask for cross-attention (decoder can attend to all encoder positions)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # Apply attention to values
        attn_output = torch.einsum('hmqk,khmd->qhmd', attn_weights, v)  # (seq_len, num_kv_heads, q_mult, head_dim)
        
        # Reshape and project
        attn_output = attn_output.reshape(seq_len, -1)  # (seq_len, num_heads * head_dim)
        output = self.out(attn_output)
        
        # Residual connection
        return residual + output


class Transformer(torch.nn.Module):
    """
    Context-aware GPT model where output from previous prediction is fed back as input.
    
    The context_state buffer stores the last token's output and is fed into each layer.
    Call reset_context() to start a new conversation.
    """
    def __init__(
        self,
        config: ModelConfig,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.config = config
        self.embedding = torch.nn.Embedding(
            config.vocab_size, config.hidden_size, device=device, dtype=torch.bfloat16
        )
        # First block is context-aware, rest are standard
        self.block = torch.nn.ModuleList()
        self.block.append(ContextTransformerBlock(config, 0, device))
        for layer_idx in range(1, config.num_hidden_layers):
            self.block.append(TransformerBlock(config, layer_idx, device))
       
        self.norm = RMSNorm(config.hidden_size, device=device)
        self.unembedding = torch.nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            device=device,
            dtype=torch.bfloat16,
        )
        
        self.register_buffer(
            'context_state',
            torch.zeros(config.hidden_size, device=device, dtype=torch.bfloat16)
        )
        
        # Learnable <start> token embedding for short sequences
        if config.use_context_embedding:
            self.start_token_embedding = torch.nn.Parameter(
                torch.randn(config.hidden_size, device=device, dtype=torch.bfloat16) * 0.02
            )

    def reset_context(self):
        """Reset context state to zeros for new conversation."""
        self.context_state = torch.zeros_like(self.context_state)

    def forward(
        self,
        x: torch.Tensor,
        update_context: bool = True,
        max_seq_len: int = None,
        encoder_k: torch.Tensor | None = None,
        encoder_v: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass with optional encoder-decoder cross-attention.
        
        Process:
        1. Process each token individually (chunk size = 1)
        2. Each token: self-attention → cross-attention (if encoder K,V provided) → MLP
        3. Update context after each token
        
        Args:
            x: Input token IDs (seq_len,)
            update_context: Whether to update context state after processing
            max_seq_len: Ignored - always process token by token
            encoder_k: Optional encoder keys (enc_seq_len, hidden_size) - SVD compressed
            encoder_v: Optional encoder values (enc_seq_len, hidden_size) - SVD compressed
        
        Note: encoder_k and encoder_v are the SVD-compressed encoder outputs, 
              used for cross-attention within each decoder token.
        """
        input_len = x.shape[0]
        
        # Process each token individually
        all_logits = []
        
        for token_idx in range(input_len):
            # Single token
            token = x[token_idx:token_idx+1]
            
            # First token gets start token, subsequent tokens use context from previous
            is_first_chunk = (token_idx == 0)
            token_logits = self._forward_chunk(
                token, 
                update_context=update_context, 
                is_first_chunk=is_first_chunk,
                encoder_k=encoder_k,
                encoder_v=encoder_v,
                return_hidden_states=False
            )
            
            all_logits.append(token_logits)
        
        # Stack all token logits
        logits = torch.cat(all_logits, dim=0)
        return logits
    
    def _forward_chunk(
        self, 
        x: torch.Tensor, 
        update_context: bool = True, 
        is_first_chunk: bool = False,
        encoder_k: torch.Tensor | None = None,
        encoder_v: torch.Tensor | None = None,
        return_hidden_states: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Process a single chunk of tokens with optional encoder cross-attention.
        
        Args:
            x: Input token IDs (chunk_size,)
            update_context: Whether to update context state
            is_first_chunk: Whether this is the first chunk
            encoder_k: Optional encoder keys (enc_seq_len, hidden_size)
            encoder_v: Optional encoder values (enc_seq_len, hidden_size)
            return_hidden_states: If True, return (logits, hidden_states) tuple
        
        Returns:
            logits (chunk_size, vocab_size) or tuple of (logits, hidden_states)
        """
        token_embeds = self.embedding(x)
        
        # Always prepend a token at the beginning of each sequence
        if self.config.use_context_embedding:
            # Use learnable <start> token only for the first chunk of a new sequence
            # For subsequent chunks, use context_state from previous chunk
            if is_first_chunk:
                # First chunk: use learnable <start> token
                prepend_embed = self.start_token_embedding.unsqueeze(0)
                context_vector = self.start_token_embedding
            else:
                # Continuation chunk: use context from previous chunk/sequence
                prepend_embed = self.context_state.unsqueeze(0)
                context_vector = self.context_state
            
            x = torch.cat([prepend_embed, token_embeds], dim=0)
            context_offset = 1  # Account for prepended token
        else:
            x = token_embeds
            context_offset = 0  # No token prepended
            context_vector = self.context_state
        
        # Pass context to first block, then use standard blocks with encoder cross-attention
        context = context_vector.unsqueeze(0).expand(x.shape[0], -1)
        
        # First block with context (ContextTransformerBlock doesn't have cross-attention)
        x = self.block[0](x, context)
        
        # Remaining blocks with optional encoder cross-attention
        for i in range(1, len(self.block)):
            x = self.block[i](x, encoder_k=encoder_k, encoder_v=encoder_v)
        
        x = self.norm(x)
        
        # Remove context embedding if it was prepended
        if context_offset > 0:
            x_tokens = x[context_offset:]
        else:
            x_tokens = x
        
        logits = self.unembedding(x_tokens)
        
        # Update context state
        if x_tokens.shape[0] > 0 and update_context:
            self.context_state = x_tokens[-1].detach().clone()
        
        if return_hidden_states:
            return logits, x_tokens
        else:
            return logits

    @staticmethod
    def from_checkpoint(
        path: str, device: str | torch.device = "cuda"
    ) -> "Transformer":
        """Load model from checkpoint directory."""
        if not isinstance(device, torch.device):
            device = torch.device(device)

        config_path = os.path.join(path, "config.json")
        with open(config_path, "r") as f:
            json_config = json.load(f)
            config = ModelConfig(**json_config)

        model = Transformer(
            config=config,
            device=device,
        )
        model.eval()

        return model
