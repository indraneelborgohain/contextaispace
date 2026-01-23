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
    """Transformer block with self-attention, context cross-attention, and MLP."""
    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        # Use standard attention instead of context-aware
        self.attn = AttentionBlock(config, layer_idx, device)
        # Add context state cross-attention
        self.context_cross_attn = ContextStateCrossAttention(config, device)
        # Use standard MLP instead of context-aware
        self.mlp = MLPBlock(config, device)

    def forward(
        self, 
        x: torch.Tensor, 
        context_state: torch.Tensor,
        encoder_k: torch.Tensor | None = None,
        encoder_v: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Self-attention
        x = self.attn(x)
        # Context cross-attention
        x = self.context_cross_attn(x, context_state)
        # MLP
        x = self.mlp(x)
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
        # No RoPE - processing one token at a time with context state for sequential info

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
        # No RoPE applied - context state handles sequential information
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
    """Standard transformer block with context cross-attention and optional encoder cross-attention."""
    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = AttentionBlock(config, layer_idx, device)
        
        # Context state cross-attention (always enabled)
        self.context_cross_attn = ContextStateCrossAttention(config, device)
        
        # Optional cross-attention to encoder
        if config.use_encoder_decoder_cross_attention:
            encoder_size = config.encoder_hidden_size if config.encoder_hidden_size else config.hidden_size
            self.encoder_cross_attn = CrossAttentionLayer(config, encoder_hidden_size=encoder_size, device=device)
        else:
            self.encoder_cross_attn = None
        
        self.mlp = MLPBlock(config, device)

    def forward(
        self, 
        x: torch.Tensor,
        context_state: torch.Tensor,
        encoder_k: torch.Tensor | None = None,
        encoder_v: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Self-attention
        x = self.attn(x)
        
        # Context cross-attention
        x = self.context_cross_attn(x, context_state)
        
        # Encoder cross-attention (if enabled and encoder K, V provided)
        if self.encoder_cross_attn is not None and encoder_k is not None and encoder_v is not None:
            x = self.encoder_cross_attn(x, encoder_k, encoder_v)
        
        # MLP
        x = self.mlp(x)
        return x


class ContextStateCrossAttention(torch.nn.Module):
    """
    Cross-attention layer for attending to context state.
    
    Q: from current token hidden states
    K, V: from context_state (previous token's hidden state)
    
    No gating needed - first token naturally has zero context.
    """
    def __init__(
        self,
        config: ModelConfig,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        
        self.norm = RMSNorm(config.hidden_size, device=device)
        
        # Q, K, V projections (all use decoder hidden size)
        self.q_proj = torch.nn.Linear(
            config.hidden_size,
            config.head_dim * config.num_attention_heads,
            device=device,
            dtype=torch.bfloat16
        )
        self.k_proj = torch.nn.Linear(
            config.hidden_size,
            config.head_dim * config.num_key_value_heads,
            device=device,
            dtype=torch.bfloat16
        )
        self.v_proj = torch.nn.Linear(
            config.hidden_size,
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
        context_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Cross-attention to context state.
        
        Args:
            x: Current token hidden states (seq_len, hidden_size)
            context_state: Previous token's hidden state (hidden_size,)
        
        Returns:
            Output (seq_len, hidden_size)
        """
        residual = x
        x = self.norm(x)
        
        seq_len = x.shape[0]
        
        # Project queries from current token
        q = self.q_proj(x)  # (seq_len, num_heads * head_dim)
        
        # Context state is a single vector, unsqueeze to make it a sequence of length 1
        context_seq = context_state.unsqueeze(0)  # (1, hidden_size)
        
        # Project keys and values from context
        k = self.k_proj(context_seq)  # (1, num_kv_heads * head_dim)
        v = self.v_proj(context_seq)  # (1, num_kv_heads * head_dim)
        
        # Reshape for multi-head attention
        q = q.view(
            seq_len,
            self.num_key_value_heads,
            self.num_attention_heads // self.num_key_value_heads,
            self.head_dim
        )  # (seq_len, num_kv_heads, q_mult, head_dim)
        
        k = k.view(1, self.num_key_value_heads, self.head_dim)
        v = v.view(1, self.num_key_value_heads, self.head_dim)
        
        # Grouped query attention - expand K and V
        q_mult = self.num_attention_heads // self.num_key_value_heads
        k = k[:, :, None, :].expand(-1, -1, q_mult, -1)  # (1, num_kv_heads, q_mult, head_dim)
        v = v[:, :, None, :].expand(-1, -1, q_mult, -1)
        
        # Compute attention scores: Q @ K^T
        attn_scores = torch.einsum('qhmd,khmd->hmqk', q, k)  # (num_kv_heads, q_mult, seq_len, 1)
        attn_scores = attn_scores * self.sm_scale
        
        # No causal mask needed (attending to single context vector)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # Apply attention to values
        attn_output = torch.einsum('hmqk,khmd->qhmd', attn_weights, v)  # (seq_len, num_kv_heads, q_mult, head_dim)
        
        # Reshape and project
        attn_output = attn_output.reshape(seq_len, -1)  # (seq_len, num_heads * head_dim)
        output = self.out(attn_output)
        
        # Residual connection (no gating - first token naturally has zero context)
        return residual + output


class EncoderCrossAttentionLayer(torch.nn.Module):
    """
    Encoder-level cross-attention layer for question attending to context.
    
    Takes raw encoder hidden states and performs cross-attention.
    Q: from question encoder hidden states
    K, V: from context encoder hidden states
    
    This layer is trainable (not frozen).
    """
    def __init__(
        self,
        encoder_hidden_size: int = 1024,
        num_attention_heads: int = 16,
        head_dim: int = 64,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.encoder_hidden_size = encoder_hidden_size
        
        self.norm_q = RMSNorm(encoder_hidden_size, device=device)
        self.norm_kv = RMSNorm(encoder_hidden_size, device=device)
        
        # Q, K, V projections (all from encoder hidden size)
        self.q_proj = torch.nn.Linear(
            encoder_hidden_size,
            head_dim * num_attention_heads,
            device=device,
            dtype=torch.bfloat16
        )
        self.k_proj = torch.nn.Linear(
            encoder_hidden_size,
            head_dim * num_attention_heads,
            device=device,
            dtype=torch.bfloat16
        )
        self.v_proj = torch.nn.Linear(
            encoder_hidden_size,
            head_dim * num_attention_heads,
            device=device,
            dtype=torch.bfloat16
        )
        
        # Output projection
        self.out = torch.nn.Linear(
            head_dim * num_attention_heads,
            encoder_hidden_size,
            device=device,
            dtype=torch.bfloat16,
        )
        
        self.sm_scale = 1 / math.sqrt(head_dim)
    
    def forward(
        self, 
        query_hidden: torch.Tensor, 
        key_value_hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Cross-attention from question to context at encoder level.
        
        Args:
            query_hidden: Question encoder hidden states (q_seq_len, encoder_hidden_size)
            key_value_hidden: Context encoder hidden states (c_seq_len, encoder_hidden_size)
        
        Returns:
            tuple: (attended_question, attended_question)
                - Returns the same attended_question for both K and V to ensure
                  they have matching sequence lengths for decoder cross-attention.
                - attended_question contains information from both question and context
                  via the cross-attention mechanism.
        """
        residual = query_hidden
        
        # Normalize
        query_hidden = self.norm_q(query_hidden)
        key_value_hidden = self.norm_kv(key_value_hidden)
        
        q_seq_len = query_hidden.shape[0]
        kv_seq_len = key_value_hidden.shape[0]
        
        # Project to Q, K, V
        q = self.q_proj(query_hidden)  # (q_seq_len, num_heads * head_dim)
        k = self.k_proj(key_value_hidden)  # (kv_seq_len, num_heads * head_dim)
        v = self.v_proj(key_value_hidden)  # (kv_seq_len, num_heads * head_dim)
        
        # Reshape for multi-head attention
        q = q.view(q_seq_len, self.num_attention_heads, self.head_dim)
        k = k.view(kv_seq_len, self.num_attention_heads, self.head_dim)
        v = v.view(kv_seq_len, self.num_attention_heads, self.head_dim)
        
        # Compute attention scores: Q @ K^T
        # q: (q_seq_len, num_heads, head_dim)
        # k: (kv_seq_len, num_heads, head_dim)
        attn_scores = torch.einsum('qhd,khd->hqk', q, k)  # (num_heads, q_seq_len, kv_seq_len)
        attn_scores = attn_scores * self.sm_scale
        
        # No causal mask for cross-attention
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # Apply attention to values
        attn_output = torch.einsum('hqk,khd->qhd', attn_weights, v)  # (q_seq_len, num_heads, head_dim)
        
        # Reshape and project
        attn_output = attn_output.reshape(q_seq_len, -1)  # (q_seq_len, num_heads * head_dim)
        output = self.out(attn_output)
        
        # Residual connection
        attended_question = residual + output
        
        # Return attended question for both K and V (it already contains context information)
        # This ensures K and V have the same sequence length for decoder cross-attention
        return attended_question, attended_question


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
        
        # Gated tanh parameter: initialized to 0 so initially no cross-attention
        # When angle=0: tanh(0)=0 → output = residual (original GPT-OSS)
        # As angle increases: tanh(angle)→1 → gradually incorporate cross-attention
        self.angle = torch.nn.Parameter(
            torch.zeros(1, device=device, dtype=torch.bfloat16)
        )
    
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
        
        # Project queries from decoder
        q = self.q_proj(x)  # (..., num_heads * head_dim)
        
        # Project keys and values from encoder
        k = self.k_proj(encoder_k)  # (..., num_kv_heads * head_dim)
        v = self.v_proj(encoder_v)  # (..., num_kv_heads * head_dim)
        
        # Infer dimensions from the projected tensors
        # Handle both batched and unbatched cases
        if q.dim() == 2:
            # Unbatched: (seq_len, num_heads * head_dim)
            seq_len = q.shape[0]
            enc_seq_len = k.shape[0]
            
            q = q.view(
                seq_len,
                self.num_key_value_heads,
                self.num_attention_heads // self.num_key_value_heads,
                self.head_dim
            )  # (seq_len, num_kv_heads, q_mult, head_dim)
            
            k = k.view(enc_seq_len, self.num_key_value_heads, self.head_dim)
            v = v.view(enc_seq_len, self.num_key_value_heads, self.head_dim)
        else:
            # Batched: (batch_size, seq_len, num_heads * head_dim)
            batch_size = q.shape[0]
            seq_len = q.shape[1]
            enc_seq_len = k.shape[1]
            
            q = q.view(
                batch_size,
                seq_len,
                self.num_key_value_heads,
                self.num_attention_heads // self.num_key_value_heads,
                self.head_dim
            )  # (batch_size, seq_len, num_kv_heads, q_mult, head_dim)
            
            k = k.view(batch_size, enc_seq_len, self.num_key_value_heads, self.head_dim)
            v = v.view(batch_size, enc_seq_len, self.num_key_value_heads, self.head_dim)
        
        # Grouped query attention
        # Expand K and V to match Q's group dimension
        q_mult = self.num_attention_heads // self.num_key_value_heads
        
        if q.dim() == 4:
            # Unbatched case
            k = k[:, :, None, :].expand(-1, -1, q_mult, -1)  # (enc_seq_len, num_kv_heads, q_mult, head_dim)
            v = v[:, :, None, :].expand(-1, -1, q_mult, -1)
            
            # Compute attention scores: Q @ K^T
            attn_scores = torch.einsum('qhmd,khmd->hmqk', q, k)  # (num_kv_heads, q_mult, seq_len, enc_seq_len)
            attn_scores = attn_scores * self.sm_scale
            
            # No causal mask for cross-attention
            attn_weights = torch.softmax(attn_scores, dim=-1)
            
            # Apply attention to values
            attn_output = torch.einsum('hmqk,khmd->qhmd', attn_weights, v)  # (seq_len, num_kv_heads, q_mult, head_dim)
            
            # Reshape and project
            seq_len = q.shape[0]
            attn_output = attn_output.reshape(seq_len, -1)  # (seq_len, num_heads * head_dim)
        else:
            # Batched case
            k = k[:, :, :, None, :].expand(-1, -1, -1, q_mult, -1)  # (batch, enc_seq_len, num_kv_heads, q_mult, head_dim)
            v = v[:, :, :, None, :].expand(-1, -1, -1, q_mult, -1)
            
            # Compute attention scores: Q @ K^T
            attn_scores = torch.einsum('bqhmd,bkhmd->bhmqk', q, k)  # (batch, num_kv_heads, q_mult, seq_len, enc_seq_len)
            attn_scores = attn_scores * self.sm_scale
            
            # No causal mask for cross-attention
            attn_weights = torch.softmax(attn_scores, dim=-1)
            
            # Apply attention to values
            attn_output = torch.einsum('bhmqk,bkhmd->bqhmd', attn_weights, v)  # (batch, seq_len, num_kv_heads, q_mult, head_dim)
            
            # Reshape and project
            batch_size = q.shape[0]
            seq_len = q.shape[1]
            attn_output = attn_output.reshape(batch_size, seq_len, -1)  # (batch, seq_len, num_heads * head_dim)
        
        output = self.out(attn_output)
        
        # Gated residual connection with tanh
        # Initially (angle=0): tanh(0)=0, so output = residual (no cross-attention)
        # As training progresses: tanh(angle) increases, gradually adding cross-attention
        gate = torch.tanh(self.angle)
        return residual + gate * output


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
        encoder_v: torch.Tensor | None = None,
        max_dec_length: int = 1
    ) -> torch.Tensor:
        """
        Forward pass with optional encoder-decoder cross-attention and chunked processing.
        
        Process:
        1. Split input into chunks of size max_dec_length
        2. Process each chunk with self-attention and cross-attention
        3. Use last token's hidden state from chunk N as context for chunk N+1
        
        Args:
            x: Input token IDs (seq_len,)
            update_context: Whether to update context state after processing
            max_seq_len: Deprecated - use max_dec_length instead
            encoder_k: Optional encoder keys (enc_seq_len, hidden_size) - SVD compressed
            encoder_v: Optional encoder values (enc_seq_len, hidden_size) - SVD compressed
            max_dec_length: Maximum chunk size for processing (default=1 for token-by-token)
        
        Note: encoder_k and encoder_v remain constant across all chunks.
        """
        input_len = x.shape[0]
        
        # Split input into chunks
        all_logits = []
        chunk_start = 0
        
        while chunk_start < input_len:
            chunk_end = min(chunk_start + max_dec_length, input_len)
            chunk = x[chunk_start:chunk_end]
            
            # First chunk uses current context_state (zeros if reset)
            # Subsequent chunks use context from previous chunk's last token
            is_first_chunk = (chunk_start == 0)
            
            chunk_logits = self._forward_chunk(
                chunk, 
                update_context=update_context, 
                is_first_chunk=is_first_chunk,
                encoder_k=encoder_k,
                encoder_v=encoder_v,
                return_hidden_states=False
            )
            
            all_logits.append(chunk_logits)
            chunk_start = chunk_end
        
        # Concatenate all chunk logits
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
        Process a single chunk of tokens with context and encoder cross-attention.
        
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
        # Embed tokens (no prepending of context)
        x = self.embedding(x)
        
        # Use context_state for cross-attention
        # First token: context_state is zeros (after reset)
        # Subsequent tokens: context_state contains previous token's hidden state
        context_state = self.context_state
        
        # Pass through all layers with context cross-attention
        # All blocks now use context cross-attention instead of prepending
        for i in range(len(self.block)):
            x = self.block[i](x, context_state=context_state, encoder_k=encoder_k, encoder_v=encoder_v)
        
        x = self.norm(x)
        logits = self.unembedding(x)
        
        # Update context state with last token's hidden state
        if x.shape[0] > 0 and update_context:
            self.context_state = x[-1].detach().clone()
        
        if return_hidden_states:
            return logits, x
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
