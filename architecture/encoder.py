"""
Bidirectional Encoder Implementation (BERT-style)

This module implements a bidirectional transformer encoder where tokens can attend
to both past and future tokens (no causal masking). Suitable for tasks like:
- Text encoding/representation learning
- Classification
- Masked language modeling (BERT-style)
"""

import math
import torch
import torch.nn as nn
from dataclasses import dataclass

from .attention_components import RMSNorm, RotaryEmbedding, _apply_rotary_emb
from .transformer import swiglu


@dataclass
class EncoderConfig:
    """Configuration for bidirectional encoder."""
    vocab_size: int = 50257
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_key_value_heads: int = 12  # Same as num_attention_heads for standard attention
    head_dim: int = 64
    intermediate_size: int = 3072
    num_experts: int = 8
    experts_per_token: int = 2
    swiglu_limit: float = 7.0
    max_position_embeddings: int = 512
    rope_theta: float = 10000.0
    rope_scaling_factor: float = 1.0
    rope_ntk_alpha: float = 1.0
    rope_ntk_beta: float = 32.0
    dropout: float = 0.1
    use_moe: bool = False  # Use Mixture of Experts or standard FFN
    use_lsi_compression: bool = False  # Use LSI cross-attention for compression instead of SVD
    num_compression_slots: int = 64  # Number of latent slots for LSI compression


def bidirectional_sdpa(Q, K, V, sm_scale, attention_mask=None):
    """
    Scaled Dot-Product Attention WITHOUT causal masking (bidirectional).
    
    Args:
        Q: Query tensor (n_tokens, n_heads, q_mult, d_head)
        K: Key tensor (n_tokens, n_heads, d_head)
        V: Value tensor (n_tokens, n_heads, d_head)
        sm_scale: Scaling factor for attention scores
        attention_mask: Optional attention mask (n_tokens, n_tokens), True = keep, False = mask
    
    Returns:
        Attention output (n_tokens, n_heads * q_mult * d_head)
    """
    n_tokens, n_heads, q_mult, d_head = Q.shape
    assert K.shape == (n_tokens, n_heads, d_head)
    assert V.shape == (n_tokens, n_heads, d_head)
    
    # Expand K and V for grouped query attention
    K = K[:, :, None, :].expand(-1, -1, q_mult, -1)
    V = V[:, :, None, :].expand(-1, -1, q_mult, -1)
    
    # Compute attention scores: Q @ K^T
    QK = torch.einsum("qhmd,khmd->hmqk", Q, K)
    QK *= sm_scale
    
    # Apply optional attention mask (but NO causal mask)
    if attention_mask is not None:
        # attention_mask should be (n_tokens, n_tokens) with True for positions to keep
        mask_value = torch.finfo(QK.dtype).min
        mask = attention_mask[None, None, :, :]  # Broadcast to (1, 1, n_tokens, n_tokens)
        QK = QK.masked_fill(~mask, mask_value)
    
    # Softmax and compute attention
    W = torch.softmax(QK, dim=-1)
    attn = torch.einsum("hmqk,khmd->qhmd", W, V)
    
    return attn.reshape(n_tokens, -1)


class BidirectionalAttentionBlock(nn.Module):
    """
    Bidirectional attention block (no causal masking).
    Allows tokens to attend to both past and future tokens.
    """
    def __init__(
        self,
        config: EncoderConfig,
        layer_idx: int = 0,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.dropout = config.dropout
        
        self.norm = RMSNorm(config.hidden_size, device=device)
        
        qkv_dim = config.head_dim * (
            config.num_attention_heads + 2 * config.num_key_value_heads
        )
        self.qkv = nn.Linear(
            config.hidden_size, qkv_dim, device=device, dtype=torch.bfloat16
        )
        self.out = nn.Linear(
            config.head_dim * config.num_attention_heads,
            config.hidden_size,
            device=device,
            dtype=torch.bfloat16,
        )
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.sm_scale = 1 / math.sqrt(config.head_dim)
        self.rope = RotaryEmbedding(
            config.head_dim,
            config.rope_theta,
            torch.float32,
            initial_context_length=config.max_position_embeddings,
            scaling_factor=config.rope_scaling_factor,
            ntk_alpha=config.rope_ntk_alpha,
            ntk_beta=config.rope_ntk_beta,
            device=device,
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass with bidirectional attention.
        
        Args:
            x: Input tensor (seq_len, hidden_size)
            attention_mask: Optional mask (seq_len, seq_len), True = keep, False = mask
        
        Returns:
            Output tensor (seq_len, hidden_size)
        """
        t = self.norm(x)
        qkv = self.qkv(t)
        
        # Split into Q, K, V
        q = qkv[:, : self.num_attention_heads * self.head_dim].contiguous()
        k = qkv[
            :,
            self.num_attention_heads * self.head_dim : 
            (self.num_attention_heads + self.num_key_value_heads) * self.head_dim,
        ].contiguous()
        v = qkv[
            :,
            (self.num_attention_heads + self.num_key_value_heads) * self.head_dim : 
            (self.num_attention_heads + 2 * self.num_key_value_heads) * self.head_dim,
        ].contiguous()

        # Reshape for attention
        q = q.view(
            -1,
            self.num_key_value_heads,
            self.num_attention_heads // self.num_key_value_heads,
            self.head_dim,
        )
        k = k.view(-1, self.num_key_value_heads, self.head_dim)
        v = v.view(-1, self.num_key_value_heads, self.head_dim)
        
        # Apply RoPE
        q, k = self.rope(q, k)
        
        # Bidirectional attention (no causal mask)
        t = bidirectional_sdpa(q, k, v, self.sm_scale, attention_mask)
        t = self.attn_dropout(t)
        t = self.out(t)
        t = self.resid_dropout(t)
        
        # Residual connection
        return x + t


class EncoderMLPBlock(nn.Module):
    """
    Standard MLP block for encoder (without MoE).
    Uses SwiGLU activation.
    """
    def __init__(
        self,
        config: EncoderConfig,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.swiglu_limit = config.swiglu_limit
        self.dropout = config.dropout
        
        self.norm = RMSNorm(config.hidden_size, device=device)
        
        # SwiGLU requires 2x intermediate size for gating
        self.fc1 = nn.Linear(
            config.hidden_size,
            config.intermediate_size * 2,
            device=device,
            dtype=torch.bfloat16
        )
        self.fc2 = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            device=device,
            dtype=torch.bfloat16
        )
        self.dropout_layer = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.norm(x)
        t = self.fc1(t)
        t = swiglu(t, limit=self.swiglu_limit)
        t = self.fc2(t)
        t = self.dropout_layer(t)
        return x + t


class LSICompressionLayer(nn.Module):
    """
    LSI Cross-Attention Compression Layer.
    Compresses variable-length encoder outputs to fixed-size representation using
    learnable latent slots that attend to the full encoder output.
    """
    def __init__(
        self,
        config: EncoderConfig,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_slots = config.num_compression_slots
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        
        # Learnable latent slots - these learn to "query" for important information
        self.latent_slots = nn.Parameter(
            torch.randn(config.num_compression_slots, config.hidden_size, device=device, dtype=torch.bfloat16) * 0.02
        )
        
        # Projection layers for cross-attention
        self.query_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            device=device,
            dtype=torch.bfloat16
        )
        self.key_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            device=device,
            dtype=torch.bfloat16
        )
        self.value_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            device=device,
            dtype=torch.bfloat16
        )
        self.out_proj = nn.Linear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            device=device,
            dtype=torch.bfloat16
        )
        
        self.norm = RMSNorm(config.hidden_size, device=device)
        self.dropout = nn.Dropout(config.dropout)
        
        self.sm_scale = 1 / math.sqrt(config.head_dim)
    
    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """
        Compress encoder output using LSI cross-attention.
        
        Args:
            encoder_output: Full encoder output (seq_len, hidden_size)
        
        Returns:
            Compressed output (num_slots, hidden_size)
        """
        seq_len, hidden_size = encoder_output.shape
        
        # Normalize encoder output
        encoder_output = self.norm(encoder_output)
        
        # Expand latent slots (no batch dimension here, single sequence)
        Q_input = self.latent_slots  # (num_slots, hidden_size)
        
        # Project to Q, K, V
        Q = self.query_proj(Q_input)  # (num_slots, num_heads * head_dim)
        K = self.key_proj(encoder_output)  # (seq_len, num_heads * head_dim)
        V = self.value_proj(encoder_output)  # (seq_len, num_heads * head_dim)
        
        # Reshape for multi-head attention
        Q = Q.view(self.num_slots, self.num_heads, self.head_dim)  # (num_slots, num_heads, head_dim)
        K = K.view(seq_len, self.num_heads, self.head_dim)  # (seq_len, num_heads, head_dim)
        V = V.view(seq_len, self.num_heads, self.head_dim)  # (seq_len, num_heads, head_dim)
        
        # Compute attention scores: Q @ K^T
        # Q: (num_slots, num_heads, head_dim)
        # K: (seq_len, num_heads, head_dim)
        scores = torch.einsum("qhd,khd->hqk", Q, K)  # (num_heads, num_slots, seq_len)
        scores = scores * self.sm_scale
        
        # Softmax over seq_len dimension (each slot attends to all encoder tokens)
        attn_weights = torch.softmax(scores, dim=-1)  # (num_heads, num_slots, seq_len)
        
        # Apply attention to values
        # attn_weights: (num_heads, num_slots, seq_len)
        # V: (seq_len, num_heads, head_dim)
        output = torch.einsum("hqk,khd->qhd", attn_weights, V)  # (num_slots, num_heads, head_dim)
        
        # Reshape and project
        output = output.reshape(self.num_slots, self.num_heads * self.head_dim)  # (num_slots, num_heads * head_dim)
        output = self.out_proj(output)  # (num_slots, hidden_size)
        output = self.dropout(output)
        
        # Residual connection with latent slots
        output = self.latent_slots + output
        
        return output


class EncoderMoEBlock(nn.Module):
    """
    Mixture of Experts MLP block for encoder.
    Uses sparse MoE with top-k expert selection.
    """
    def __init__(
        self,
        config: EncoderConfig,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.num_experts = config.num_experts
        self.experts_per_token = config.experts_per_token
        self.swiglu_limit = config.swiglu_limit
        self.dropout = config.dropout
        
        self.norm = RMSNorm(config.hidden_size, device=device)
        
        self.gate = nn.Linear(
            config.hidden_size, config.num_experts, device=device, dtype=torch.bfloat16
        )
        
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(
                    config.hidden_size, 
                    config.intermediate_size * 2, 
                    device=device, 
                    dtype=torch.bfloat16
                ),
                nn.Linear(
                    config.intermediate_size, 
                    config.hidden_size, 
                    device=device, 
                    dtype=torch.bfloat16
                )
            ) for _ in range(config.num_experts)
        ])
        self.dropout_layer = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len, hidden_size = x.shape
        
        t = self.norm(x)
        g = self.gate(t)
        
        # Top-k expert selection
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
            
            # Apply expert
            expert_out = expert_input
            expert_out = self.experts[expert_idx][0](expert_out)
            expert_out = swiglu(expert_out, limit=self.swiglu_limit)
            expert_out = self.experts[expert_idx][1](expert_out)
            
            output[token_indices] += expert_out * weights.unsqueeze(-1)
        
        output = output.view(seq_len, hidden_size)
        output = self.dropout_layer(output)
        
        return x + output


class EncoderBlock(nn.Module):
    """Bidirectional transformer encoder block."""
    def __init__(
        self,
        config: EncoderConfig,
        layer_idx: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = BidirectionalAttentionBlock(config, layer_idx, device)
        
        if config.use_moe:
            self.mlp = EncoderMoEBlock(config, device)
        else:
            self.mlp = EncoderMLPBlock(config, device)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass through encoder block.
        
        Args:
            x: Input tensor (seq_len, hidden_size)
            attention_mask: Optional attention mask (seq_len, seq_len)
        
        Returns:
            Output tensor (seq_len, hidden_size)
        """
        x = self.attn(x, attention_mask)
        x = self.mlp(x)
        return x


class BidirectionalEncoder(nn.Module):
    """
    Bidirectional Transformer Encoder (BERT-style).
    
    Encodes input sequences with full bidirectional attention.
    Suitable for:
    - Representation learning
    - Classification tasks
    - Masked language modeling
    
    Supports chunking for long sequences with context embedding propagation.
    Supports LSI cross-attention compression for fixed-size encoder output.
    """
    def __init__(
        self,
        config: EncoderConfig,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(
            config.vocab_size, config.hidden_size, device=device, dtype=torch.bfloat16
        )
        self.dropout = nn.Dropout(config.dropout)
        
        # Stack of encoder blocks
        self.blocks = nn.ModuleList([
            EncoderBlock(config, layer_idx, device)
            for layer_idx in range(config.num_hidden_layers)
        ])
        
        self.norm = RMSNorm(config.hidden_size, device=device)
        
        # LSI compression layer (optional)
        if config.use_lsi_compression:
            self.lsi_compression = LSICompressionLayer(config, device=device)
        else:
            self.lsi_compression = None
        
        # Learnable start token embedding for chunking
        self.start_token_embedding = nn.Parameter(
            torch.randn(config.hidden_size, device=device, dtype=torch.bfloat16) * 0.02
        )
        
        # Buffer for context from previous chunk
        self.register_buffer(
            'chunk_context',
            torch.zeros(config.hidden_size, device=device, dtype=torch.bfloat16)
        )
        
        self.norm = RMSNorm(config.hidden_size, device=device)
        
        # Learnable start token embedding for chunking
        self.start_token_embedding = nn.Parameter(
            torch.randn(config.hidden_size, device=device, dtype=torch.bfloat16) * 0.02
        )
        
        # Buffer for context from previous chunk
        self.register_buffer(
            'chunk_context',
            torch.zeros(config.hidden_size, device=device, dtype=torch.bfloat16)
        )
    
    def reset_context(self):
        """Reset chunk context to zeros."""
        self.chunk_context = torch.zeros_like(self.chunk_context)
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor | None = None,
        return_context_embeddings: bool = False,
        return_compressed_kv: bool = False,
        chunk_size: int | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input tokens with bidirectional attention.
        
        Args:
            input_ids: Input token IDs (seq_len,)
            attention_mask: Optional mask (seq_len,) where 1 = real token, 0 = padding
            return_context_embeddings: If True, returns list of context embeddings (one per chunk)
            return_compressed_kv: If True, returns SVD-compressed K, V for cross-attention
            chunk_size: Chunk size for processing long sequences. If None, uses max_position_embeddings
        
        Returns:
            If return_context_embeddings=False: Encoded representations (seq_len, hidden_size)
            If return_context_embeddings=True: Tensor of context embeddings (num_chunks, hidden_size)
            If return_compressed_kv=True: Tuple of (K_compressed, V_compressed) each (chunk_size, hidden_size)
        """
        if chunk_size is None:
            chunk_size = self.config.max_position_embeddings
        
        input_len = input_ids.shape[0]
        
        # If SVD compression requested, always use chunking
        if return_compressed_kv:
            return self._forward_with_svd_compression(input_ids, attention_mask, chunk_size)
        
        # If chunking is needed
        if input_len > chunk_size or return_context_embeddings:
            return self._forward_chunked(input_ids, attention_mask, chunk_size, return_context_embeddings)
        
        # Normal processing without chunking
        # Embed tokens
        x = self.embedding(input_ids)
        x = self.dropout(x)
        
        # Create attention mask if provided
        # Convert from (seq_len,) to (seq_len, seq_len)
        if attention_mask is not None:
            # attention_mask: (seq_len,) with 1 for real tokens, 0 for padding
            # Create pairwise mask: (seq_len, seq_len)
            seq_len = attention_mask.shape[0]
            attn_mask_2d = attention_mask.unsqueeze(0) & attention_mask.unsqueeze(1)
        else:
            attn_mask_2d = None
        
        # Pass through encoder blocks
        for block in self.blocks:
            x = block(x, attn_mask_2d)
        
        # Final normalization
        x = self.norm(x)
        
        return x
    
    def _forward_chunked(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        chunk_size: int,
        return_context_embeddings: bool
    ) -> torch.Tensor:
        """
        Process input in chunks and collect context embeddings.
        
        Returns:
            If return_context_embeddings=True: Tensor of context embeddings (num_chunks, hidden_size)
            If return_context_embeddings=False: Concatenated outputs (seq_len, hidden_size)
        """
        input_len = input_ids.shape[0]
        context_embeddings = []
        all_outputs = []
        
        # Calculate number of chunks (pad last chunk to chunk_size if needed)
        num_chunks = (input_len + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, input_len)
            
            # Get chunk tokens
            chunk_tokens = input_ids[start_idx:end_idx]
            chunk_len = chunk_tokens.shape[0]
            
            # Pad chunk if it's the last one and shorter than chunk_size
            if chunk_len < chunk_size:
                # Pad with zeros (NULL tokens)
                padding = torch.zeros(
                    chunk_size - chunk_len, 
                    dtype=chunk_tokens.dtype, 
                    device=chunk_tokens.device
                )
                chunk_tokens = torch.cat([chunk_tokens, padding], dim=0)
                
                # Update attention mask if provided
                if attention_mask is not None:
                    chunk_mask = attention_mask[start_idx:end_idx]
                    mask_padding = torch.zeros(
                        chunk_size - chunk_len,
                        dtype=chunk_mask.dtype,
                        device=chunk_mask.device
                    )
                    chunk_mask = torch.cat([chunk_mask, mask_padding], dim=0)
                else:
                    chunk_mask = None
            else:
                chunk_mask = attention_mask[start_idx:end_idx] if attention_mask is not None else None
            
            # Process chunk
            chunk_output = self._forward_chunk(
                chunk_tokens, 
                chunk_mask, 
                is_first_chunk=(chunk_idx == 0)
            )
            
            # Extract context embedding (last token's hidden state before padding)
            # Use the actual last token position before padding
            context_idx = chunk_len - 1 if chunk_len < chunk_size else chunk_size - 1
            context_embedding = chunk_output[context_idx].detach().clone()
            context_embeddings.append(context_embedding)
            
            # Update chunk context for next iteration
            self.chunk_context = context_embedding
            
            if not return_context_embeddings:
                # Only keep outputs for actual tokens (not padding)
                if chunk_len < chunk_size:
                    all_outputs.append(chunk_output[:chunk_len])
                else:
                    all_outputs.append(chunk_output)
        
        if return_context_embeddings:
            # Stack context embeddings: (num_chunks, hidden_size)
            return torch.stack(context_embeddings, dim=0)
        else:
            # Concatenate all chunk outputs
            return torch.cat(all_outputs, dim=0)
    
    def _forward_chunk(
        self,
        chunk_tokens: torch.Tensor,
        chunk_mask: torch.Tensor | None,
        is_first_chunk: bool
    ) -> torch.Tensor:
        """Process a single chunk with context prepending."""
        # Embed tokens
        token_embeds = self.embedding(chunk_tokens)
        
        # Prepend context embedding
        if is_first_chunk:
            prepend_embed = self.start_token_embedding.unsqueeze(0)
        else:
            prepend_embed = self.chunk_context.unsqueeze(0)
        
        x = torch.cat([prepend_embed, token_embeds], dim=0)
        x = self.dropout(x)
        
        # Create attention mask
        if chunk_mask is not None:
            # Add mask for prepended token (always attend to it)
            prepend_mask = torch.ones(1, dtype=chunk_mask.dtype, device=chunk_mask.device)
            full_mask = torch.cat([prepend_mask, chunk_mask], dim=0)
            
            # Create pairwise mask
            seq_len = full_mask.shape[0]
            attn_mask_2d = full_mask.unsqueeze(0) & full_mask.unsqueeze(1)
        else:
            attn_mask_2d = None
        
        # Pass through encoder blocks
        for block in self.blocks:
            x = block(x, attn_mask_2d)
        
        x = self.norm(x)
        
        # Remove prepended token
        x = x[1:]
        
        return x
    
    def _forward_with_svd_compression(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        chunk_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Process input in chunks, stack outputs, and compress with SVD.
        
        Returns:
            Tuple of (K_compressed, V_compressed) each (chunk_size, hidden_size)
        """
        input_len = input_ids.shape[0]
        all_outputs = []
        
        # Calculate number of chunks
        num_chunks = (input_len + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, input_len)
            
            # Get chunk tokens
            chunk_tokens = input_ids[start_idx:end_idx]
            chunk_len = chunk_tokens.shape[0]
            
            # Pad chunk if needed
            if chunk_len < chunk_size:
                padding = torch.zeros(
                    chunk_size - chunk_len, 
                    dtype=chunk_tokens.dtype, 
                    device=chunk_tokens.device
                )
                chunk_tokens = torch.cat([chunk_tokens, padding], dim=0)
                
                if attention_mask is not None:
                    chunk_mask = attention_mask[start_idx:end_idx]
                    mask_padding = torch.zeros(
                        chunk_size - chunk_len,
                        dtype=chunk_mask.dtype,
                        device=chunk_mask.device
                    )
                    chunk_mask = torch.cat([chunk_mask, mask_padding], dim=0)
                else:
                    chunk_mask = None
            else:
                chunk_mask = attention_mask[start_idx:end_idx] if attention_mask is not None else None
            
            # Process chunk
            chunk_output = self._forward_chunk(
                chunk_tokens, 
                chunk_mask, 
                is_first_chunk=(chunk_idx == 0)
            )
            
            # Only keep actual tokens (not padding)
            if chunk_len < chunk_size:
                chunk_output = chunk_output[:chunk_len]
            
            all_outputs.append(chunk_output)
            
            # Update context for next chunk
            self.chunk_context = chunk_output[-1].detach().clone()
        
        # Stack all outputs: (total_tokens, hidden_size)
        stacked_outputs = torch.cat(all_outputs, dim=0)  # (seq_len, hidden_size)
        
        # Apply compression
        if self.config.use_lsi_compression:
            # Use LSI cross-attention for learned compression
            compressed = self.lsi_compression(stacked_outputs)  # (num_slots, hidden_size)
            K_compressed = compressed
            V_compressed = compressed
        else:
            # Use SVD for deterministic compression
            K_compressed = self._compress_with_svd(stacked_outputs, chunk_size)  # (chunk_size, hidden_size)
            V_compressed = self._compress_with_svd(stacked_outputs, chunk_size)  # (chunk_size, hidden_size)
        
        return K_compressed, V_compressed
    
    def _compress_with_svd(self, X: torch.Tensor, n_components: int) -> torch.Tensor:
        """
        Compress tensor using SVD to keep top n_components.
        
        Args:
            X: Input tensor (seq_len, hidden_size)
            n_components: Number of components to keep (target seq_len)
        
        Returns:
            Compressed tensor (n_components, hidden_size)
        """
        seq_len, hidden_size = X.shape
        
        # If already smaller or equal, just return (possibly padded)
        if seq_len <= n_components:
            if seq_len < n_components:
                # Pad with zeros
                padding = torch.zeros(
                    n_components - seq_len,
                    hidden_size,
                    dtype=X.dtype,
                    device=X.device
                )
                return torch.cat([X, padding], dim=0)
            return X
        
        # Perform SVD: X = U @ S @ V^T
        # X: (seq_len, hidden_size)
        # We want to reduce seq_len dimension
        U, S, Vt = torch.svd(X)  # U: (seq_len, seq_len), S: (min(seq_len, hidden_size),), Vt: (hidden_size, hidden_size)
        
        # Keep top n_components
        U_reduced = U[:, :n_components]  # (seq_len, n_components)
        S_reduced = S[:n_components]  # (n_components,)
        
        # Reconstruct with reduced dimensions
        # We want output of shape (n_components, hidden_size)
        # X_compressed = U_reduced.T @ X
        X_compressed = U_reduced.T @ X  # (n_components, hidden_size)
        
        return X_compressed
    
    def get_pooled_output(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor | None = None,
        pooling: str = "mean"
    ) -> torch.Tensor:
        """
        Get pooled representation of the sequence.
        
        Args:
            input_ids: Input token IDs (seq_len,)
            attention_mask: Optional mask (seq_len,)
            pooling: Pooling strategy - "mean", "max", or "first" (CLS token)
        
        Returns:
            Pooled vector (hidden_size,)
        """
        x = self.forward(input_ids, attention_mask)
        
        if pooling == "first":
            # Use first token (like BERT's [CLS])
            return x[0]
        elif pooling == "max":
            # Max pooling over sequence
            if attention_mask is not None:
                # Mask out padding tokens
                mask = attention_mask.unsqueeze(-1).to(x.dtype)
                x = x * mask + (1 - mask) * (-1e9)
            return x.max(dim=0)[0]
        else:  # mean pooling
            # Mean pooling over sequence
            if attention_mask is not None:
                # Only average over real tokens
                mask = attention_mask.unsqueeze(-1).to(x.dtype)
                sum_x = (x * mask).sum(dim=0)
                count = mask.sum(dim=0).clamp(min=1)
                return sum_x / count
            else:
                return x.mean(dim=0)


class EncoderForMaskedLM(nn.Module):
    """
    Encoder with masked language modeling head (BERT-style).
    """
    def __init__(
        self,
        config: EncoderConfig,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.encoder = BidirectionalEncoder(config, device)
        
        # MLM prediction head
        self.mlm_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size, device=device, dtype=torch.bfloat16),
            nn.GELU(),
            RMSNorm(config.hidden_size, device=device),
            nn.Linear(config.hidden_size, config.vocab_size, device=device, dtype=torch.bfloat16)
        )
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass for masked language modeling.
        
        Args:
            input_ids: Input token IDs with [MASK] tokens (seq_len,)
            attention_mask: Optional mask (seq_len,)
        
        Returns:
            Logits for all positions (seq_len, vocab_size)
        """
        # Encode
        x = self.encoder(input_ids, attention_mask)
        
        # Predict tokens
        logits = self.mlm_head(x)
        
        return logits


class EncoderForClassification(nn.Module):
    """
    Encoder with classification head.
    """
    def __init__(
        self,
        config: EncoderConfig,
        num_classes: int,
        pooling: str = "first",
        device: torch.device | None = None,
    ):
        super().__init__()
        self.encoder = BidirectionalEncoder(config, device)
        self.pooling = pooling
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, num_classes, device=device, dtype=torch.bfloat16)
        )
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass for classification.
        
        Args:
            input_ids: Input token IDs (seq_len,)
            attention_mask: Optional mask (seq_len,)
        
        Returns:
            Classification logits (num_classes,)
        """
        # Get pooled representation
        pooled = self.encoder.get_pooled_output(input_ids, attention_mask, self.pooling)
        
        # Classify
        logits = self.classifier(pooled)
        
        return logits
