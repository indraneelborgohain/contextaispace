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
    Supports cross-attention where Q and K/V can have different sequence lengths.
    Handles both batched and unbatched inputs.
    
    Args:
        Q: Query tensor - unbatched: (q_len, n_heads, q_mult, d_head) or batched: (batch_size, q_len, n_heads, q_mult, d_head)
        K: Key tensor - unbatched: (k_len, n_heads, d_head) or batched: (batch_size, k_len, n_heads, d_head)
        V: Value tensor - unbatched: (k_len, n_heads, d_head) or batched: (batch_size, k_len, n_heads, d_head)
        sm_scale: Scaling factor for attention scores
        attention_mask: Optional attention mask (q_len, k_len) or (batch_size, q_len, k_len), True = keep, False = mask
    
    Returns:
        Attention output - unbatched: (q_len, n_heads * q_mult * d_head) or batched: (batch_size, q_len, n_heads * q_mult * d_head)
    """
    if Q.dim() == 4:
        # Unbatched case
        q_len, n_heads, q_mult, d_head = Q.shape
        k_len = K.shape[0]
        assert K.shape == (k_len, n_heads, d_head), f"K shape mismatch: expected ({k_len}, {n_heads}, {d_head}), got {K.shape}"
        assert V.shape == (k_len, n_heads, d_head), f"V shape mismatch: expected ({k_len}, {n_heads}, {d_head}), got {V.shape}"
        
        # Expand K and V for grouped query attention
        K = K[:, :, None, :].expand(-1, -1, q_mult, -1)
        V = V[:, :, None, :].expand(-1, -1, q_mult, -1)
        
        # Compute attention scores: Q @ K^T
        QK = torch.einsum("qhmd,khmd->hmqk", Q, K)
        QK *= sm_scale
        
        # Apply optional attention mask (but NO causal mask)
        if attention_mask is not None:
            # attention_mask should be (q_len, k_len) with True for positions to keep
            mask_value = torch.finfo(QK.dtype).min
            mask = attention_mask[None, None, :, :]  # Broadcast to (1, 1, q_len, k_len)
            QK = QK.masked_fill(~mask, mask_value)
        
        # Softmax and compute attention
        W = torch.softmax(QK, dim=-1)
        attn = torch.einsum("hmqk,khmd->qhmd", W, V)
        
        return attn.reshape(q_len, -1)
    else:
        # Batched case
        batch_size, q_len, n_heads, q_mult, d_head = Q.shape
        k_len = K.shape[1]
        assert K.shape == (batch_size, k_len, n_heads, d_head), f"K shape mismatch: expected ({batch_size}, {k_len}, {n_heads}, {d_head}), got {K.shape}"
        assert V.shape == (batch_size, k_len, n_heads, d_head), f"V shape mismatch: expected ({batch_size}, {k_len}, {n_heads}, {d_head}), got {V.shape}"
        
        # Expand K and V for grouped query attention
        K = K[:, :, :, None, :].expand(-1, -1, -1, q_mult, -1)
        V = V[:, :, :, None, :].expand(-1, -1, -1, q_mult, -1)
        
        # Compute attention scores: Q @ K^T
        QK = torch.einsum("bqhmd,bkhmd->bhmqk", Q, K)
        QK *= sm_scale
        
        # Apply optional attention mask (but NO causal mask)
        if attention_mask is not None:
            # attention_mask should be (batch_size, q_len, k_len) with True for positions to keep
            mask_value = torch.finfo(QK.dtype).min
            mask = attention_mask[:, None, None, :, :]  # Broadcast to (batch_size, 1, 1, q_len, k_len)
            QK = QK.masked_fill(~mask, mask_value)
        
        # Softmax and compute attention
        W = torch.softmax(QK, dim=-1)
        attn = torch.einsum("bhmqk,bkhmd->bqhmd", W, V)
        
        return attn.reshape(batch_size, q_len, -1)


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
        Handles both batched and unbatched inputs.
        
        Args:
            x: Input tensor - unbatched: (seq_len, hidden_size) or batched: (batch_size, seq_len, hidden_size)
            attention_mask: Optional mask - unbatched: (seq_len, seq_len) or batched: (batch_size, seq_len, seq_len), True = keep, False = mask
        
        Returns:
            Output tensor - same shape as input
        """
        t = self.norm(x)
        qkv = self.qkv(t)
        
        # Determine if batched or unbatched
        is_batched = x.dim() == 3
        
        if is_batched:
            batch_size, seq_len = x.shape[0], x.shape[1]
            
            # Split into Q, K, V
            q = qkv[:, :, : self.num_attention_heads * self.head_dim].contiguous()
            k = qkv[
                :, :,
                self.num_attention_heads * self.head_dim : 
                (self.num_attention_heads + self.num_key_value_heads) * self.head_dim,
            ].contiguous()
            v = qkv[
                :, :,
                (self.num_attention_heads + self.num_key_value_heads) * self.head_dim : 
                (self.num_attention_heads + 2 * self.num_key_value_heads) * self.head_dim,
            ].contiguous()

            # Reshape for attention
            q = q.view(
                batch_size,
                seq_len,
                self.num_key_value_heads,
                self.num_attention_heads // self.num_key_value_heads,
                self.head_dim,
            )
            k = k.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
            v = v.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        else:
            # Unbatched case
            seq_len = x.shape[0]
            
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
                seq_len,
                self.num_key_value_heads,
                self.num_attention_heads // self.num_key_value_heads,
                self.head_dim,
            )
            k = k.view(seq_len, self.num_key_value_heads, self.head_dim)
            v = v.view(seq_len, self.num_key_value_heads, self.head_dim)
        
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
        """Forward pass supporting both batched and unbatched inputs."""
        original_shape = x.shape
        is_batched = x.dim() == 3
        
        if is_batched:
            batch_size, seq_len, hidden_size = x.shape
        else:
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
        
        output = output.view(*original_shape)
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
        Handles both batched and unbatched inputs.
        
        Args:
            x: Input tensor - unbatched: (seq_len, hidden_size) or batched: (batch_size, seq_len, hidden_size)
            attention_mask: Optional attention mask - unbatched: (seq_len, seq_len) or batched: (batch_size, seq_len, seq_len)
        
        Returns:
            Output tensor - same shape as input
        """
        x = self.attn(x, attention_mask)
        x = self.mlp(x)
        return x


class BidirectionalEncoder(nn.Module):
    """
    Bidirectional Transformer Encoder with Reverse-Order Chunk Processing.
    
    Processes long sequences by:
    1. Chunking in reverse order (last chunk first)
    2. Cross-attention between consecutive chunks
    3. Final cross-attention from first to last chunk
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
        
        # Cross-attention layer for inter-chunk attention
        self.cross_attn = BidirectionalAttentionBlock(config, layer_idx=0, device=device)
        
        # Final cross-attention layer
        self.final_cross_attn = BidirectionalAttentionBlock(config, layer_idx=0, device=device)
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor | None = None,
        return_encoder_kv: bool = False,
        return_hidden_states: bool = False,
        sequence_length: int | None = None,
        sep_token_id: int | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input tokens with chunk processing and SVD compression.
        
        Args:
            input_ids: Input token IDs (total_sequence_length,)
            attention_mask: Optional mask (total_sequence_length,) where 1 = real token, 0 = padding
            return_encoder_kv: If True, returns (encoder_key, encoder_value) for decoder cross-attention
            return_hidden_states: If True, returns raw hidden states without compression (for encoder cross-attention)
            sequence_length: Maximum sequence length per chunk. If None, uses max_position_embeddings
            sep_token_id: Token ID for <SEP> to split context and question. 
                         If provided, tokens before <SEP> = context (compressed), after <SEP> = question (query).
                         If None, falls back to last-chunk-as-question logic.
        
        Returns:
            If return_encoder_kv=False and return_hidden_states=False: Encoded representations (total_sequence_length, hidden_size)
            If return_encoder_kv=True: Tuple of (encoder_key, encoder_value) each (question_length, hidden_size)
            If return_hidden_states=True: Raw hidden states (total_sequence_length, hidden_size) - no compression
        """
        if sequence_length is None:
            sequence_length = self.config.max_position_embeddings
        
        total_length = input_ids.shape[0]
        
        # Single chunk case - no chunking needed
        if total_length <= sequence_length and sep_token_id is None:
            x = self._process_single_chunk(input_ids, attention_mask)
            if return_hidden_states:
                # Return raw hidden states (for encoder cross-attention)
                return x
            if return_encoder_kv:
                # For single chunk, just return the output as both K and V
                return x, x
            return x
        
        # Multi-chunk case or <SEP> splitting
        if return_hidden_states:
            # Return raw hidden states without compression
            x = self._process_single_chunk(input_ids, attention_mask)
            return x
        if return_encoder_kv:
            return self._forward_with_chunking(input_ids, attention_mask, sequence_length, sep_token_id)
        else:
            # Just return the encoded representation without K,V extraction
            encoder_k, encoder_v = self._forward_with_chunking(input_ids, attention_mask, sequence_length, sep_token_id)
            return encoder_k
    
    def _process_single_chunk(
        self,
        chunk_tokens: torch.Tensor,
        chunk_mask: torch.Tensor | None
    ) -> torch.Tensor:
        """
        Process a single chunk through transformer blocks.
        Handles both batched and unbatched inputs.
        
        Args:
            chunk_tokens: Token IDs - unbatched: (chunk_length,) or batched: (batch_size, chunk_length)
            chunk_mask: Attention mask - unbatched: (chunk_length,) or batched: (batch_size, chunk_length) or None
        
        Returns:
            Encoded output - same batch structure as input
        """
        # Embed tokens
        x = self.embedding(chunk_tokens)
        x = self.dropout(x)
        
        # Create attention mask if provided
        if chunk_mask is not None:
            if chunk_mask.dim() == 1:
                # Unbatched: (seq_len,) -> (seq_len, seq_len)
                seq_len = chunk_mask.shape[0]
                attn_mask_2d = chunk_mask.unsqueeze(0) & chunk_mask.unsqueeze(1)
            else:
                # Batched: (batch_size, seq_len) -> (batch_size, seq_len, seq_len)
                batch_size, seq_len = chunk_mask.shape
                attn_mask_2d = chunk_mask.unsqueeze(1) & chunk_mask.unsqueeze(2)
        else:
            attn_mask_2d = None
        
        # Pass through encoder blocks
        for block in self.blocks:
            x = block(x, attn_mask_2d)
        
        # Final normalization
        x = self.norm(x)
        
        return x
    
    def _forward_with_chunking(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        sequence_length: int,
        sep_token_id: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Process input in chunks with sequential cross-attention (no SVD).
        
        Strategy:
        - Chunk 1: Process with self-attention only
        - Chunk 2: Q from chunk 2 attends to K,V from chunk 1
        - Chunk 3: Q from chunk 3 attends to K,V from chunk 2
        - etc.
        - Return final chunk's output as encoder K,V
        
        Args:
            input_ids: Input token IDs (total_sequence_length,)
            attention_mask: Optional mask (total_sequence_length,)
            sequence_length: Target length for each chunk
            sep_token_id: Ignored in new implementation
        
        Returns:
            Tuple of (encoder_key, encoder_value) from final chunk
        """
        total_length = input_ids.shape[0]
        
        # Handle single chunk case
        if total_length <= sequence_length:
            chunk_output = self._process_single_chunk(input_ids, attention_mask)
            # Return output as both K and V
            return chunk_output, chunk_output
        
        # Multi-chunk sequential cross-attention
        chunks = self._create_chunks(total_length, sequence_length)
        
        # Process first chunk (self-attention only)
        start_idx, end_idx, actual_length = chunks[0]
        chunk_tokens = input_ids[start_idx:end_idx]
        chunk_mask = attention_mask[start_idx:end_idx] if attention_mask is not None else None
        
        # Pad first chunk if needed
        if chunk_tokens.shape[0] < sequence_length:
            padding_len = sequence_length - chunk_tokens.shape[0]
            chunk_tokens = torch.cat([
                chunk_tokens,
                torch.zeros(padding_len, dtype=chunk_tokens.dtype, device=chunk_tokens.device)
            ])
            if chunk_mask is not None:
                chunk_mask = torch.cat([
                    chunk_mask,
                    torch.zeros(padding_len, dtype=chunk_mask.dtype, device=chunk_mask.device)
                ])
        
        prev_output = self._process_single_chunk(chunk_tokens, chunk_mask)
        prev_K, prev_V = prev_output[:actual_length], prev_output[:actual_length]  # Only keep actual tokens
        
        # Process remaining chunks with cross-attention to previous chunk
        for i in range(1, len(chunks)):
            start_idx, end_idx, actual_length = chunks[i]
            chunk_tokens = input_ids[start_idx:end_idx]
            chunk_mask = attention_mask[start_idx:end_idx] if attention_mask is not None else None
            
            # Pad if needed
            if chunk_tokens.shape[0] < sequence_length:
                padding_len = sequence_length - chunk_tokens.shape[0]
                chunk_tokens = torch.cat([
                    chunk_tokens,
                    torch.zeros(padding_len, dtype=chunk_tokens.dtype, device=chunk_tokens.device)
                ])
                if chunk_mask is not None:
                    chunk_mask = torch.cat([
                        chunk_mask,
                        torch.zeros(padding_len, dtype=chunk_mask.dtype, device=chunk_mask.dtype)
                    ])
            
            # Process current chunk
            curr_output = self._process_single_chunk(chunk_tokens, chunk_mask)
            
            # Extract Q from current chunk
            Q_curr, _, _ = self._extract_qkv(curr_output[:actual_length], chunk_mask[:actual_length] if chunk_mask is not None else None)
            
            # Apply cross-attention: Q_curr attends to K,V from previous chunk
            attended_output, _, _ = self._apply_cross_attention(Q_curr, prev_K, prev_V, None)
            
            # Update prev_K, prev_V to current chunk's attended output
            prev_K = attended_output
            prev_V = attended_output
        
        # Return final chunk's representation as both K and V
        return prev_K, prev_V
    
    def _forward_with_chunking_legacy(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        sequence_length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Legacy chunking: last chunk = question, other chunks = context.
        Used when no <SEP> token is found.
        """
        total_length = input_ids.shape[0]
        chunks = self._create_chunks(total_length, sequence_length)
        num_chunks = len(chunks)
        
        # If only one chunk, process normally without compression
        if num_chunks == 1:
            start_idx, end_idx, actual_length = chunks[0]
            chunk_tokens = input_ids[start_idx:end_idx]
            chunk_mask = attention_mask[start_idx:end_idx] if attention_mask is not None else None
            chunk_output = self._process_single_chunk(chunk_tokens, chunk_mask)
            Q, K, V = self._extract_qkv(chunk_output, chunk_mask)
            return K[:actual_length], V[:actual_length]
        
        # Process the LAST chunk (query source)
        last_chunk_idx = num_chunks - 1
        last_start, last_end, last_actual = chunks[last_chunk_idx]
        last_chunk_tokens = input_ids[last_start:last_end]
        last_chunk_mask = attention_mask[last_start:last_end] if attention_mask is not None else None
        last_chunk_output = self._process_single_chunk(last_chunk_tokens, last_chunk_mask)
        Q_last, _, _ = self._extract_qkv(last_chunk_output, last_chunk_mask)
        
        # Process all OTHER chunks and collect their K, V
        all_K = []
        all_V = []
        
        for idx in range(num_chunks - 1):
            start_idx, end_idx, actual_length = chunks[idx]
            chunk_tokens = input_ids[start_idx:end_idx]
            chunk_mask = attention_mask[start_idx:end_idx] if attention_mask is not None else None
            chunk_output = self._process_single_chunk(chunk_tokens, chunk_mask)
            _, K_chunk, V_chunk = self._extract_qkv(chunk_output, chunk_mask)
            all_K.append(K_chunk[:actual_length])
            all_V.append(V_chunk[:actual_length])
        
        stacked_K = torch.cat(all_K, dim=0)
        stacked_V = torch.cat(all_V, dim=0)
        compressed_K = self._compress_with_svd(stacked_K, sequence_length)
        compressed_V = self._compress_with_svd(stacked_V, sequence_length)
        
        encoder_key, encoder_value = self._apply_final_cross_attention(
            Q_last[:last_actual], compressed_K, compressed_V
        )
        
        return encoder_key, encoder_value
    
    def _process_question_only(
        self,
        question_tokens: torch.Tensor,
        question_mask: torch.Tensor | None,
        sequence_length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process when there's no context, just question."""
        question_length = question_tokens.shape[0]
        question_chunks = self._create_chunks(question_length, sequence_length)
        all_outputs = []
        
        for start_idx, end_idx, actual_length in question_chunks:
            chunk_tokens = question_tokens[start_idx:end_idx]
            chunk_mask = question_mask[start_idx:end_idx] if question_mask is not None else None
            
            if chunk_tokens.shape[0] < sequence_length:
                padding_len = sequence_length - chunk_tokens.shape[0]
                chunk_tokens = torch.cat([
                    chunk_tokens,
                    torch.zeros(padding_len, dtype=chunk_tokens.dtype, device=chunk_tokens.device)
                ])
            
            chunk_output = self._process_single_chunk(chunk_tokens, chunk_mask)
            _, K, V = self._extract_qkv(chunk_output, chunk_mask)
            all_outputs.append((K[:actual_length], V[:actual_length]))
        
        all_K = torch.cat([o[0] for o in all_outputs], dim=0)
        all_V = torch.cat([o[1] for o in all_outputs], dim=0)
        return all_K, all_V
    
    def _process_context_only(
        self,
        context_tokens: torch.Tensor,
        context_mask: torch.Tensor | None,
        sequence_length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process when there's no question, just context."""
        context_length = context_tokens.shape[0]
        context_chunks = self._create_chunks(context_length, sequence_length)
        all_K = []
        all_V = []
        
        for start_idx, end_idx, actual_length in context_chunks:
            chunk_tokens = context_tokens[start_idx:end_idx]
            chunk_mask = context_mask[start_idx:end_idx] if context_mask is not None else None
            
            if chunk_tokens.shape[0] < sequence_length:
                padding_len = sequence_length - chunk_tokens.shape[0]
                chunk_tokens = torch.cat([
                    chunk_tokens,
                    torch.zeros(padding_len, dtype=chunk_tokens.dtype, device=chunk_tokens.device)
                ])
            
            chunk_output = self._process_single_chunk(chunk_tokens, chunk_mask)
            _, K, V = self._extract_qkv(chunk_output, chunk_mask)
            all_K.append(K[:actual_length])
            all_V.append(V[:actual_length])
        
        stacked_K = torch.cat(all_K, dim=0)
        stacked_V = torch.cat(all_V, dim=0)
        
        # Compress to sequence_length
        compressed_K = self._compress_with_svd(stacked_K, sequence_length)
        compressed_V = self._compress_with_svd(stacked_V, sequence_length)
        return compressed_K, compressed_V
    
    def _create_chunks(
        self,
        total_length: int,
        sequence_length: int
    ) -> list[tuple[int, int, int]]:
        """
        Create chunk boundaries ensuring each chunk is exactly sequence_length.
        
        Args:
            total_length: Total number of tokens
            sequence_length: Target length for each chunk
        
        Returns:
            List of (start_idx, end_idx, actual_token_count) tuples
            
        Example with 210 tokens and sequence_length=100:
            - Chunk 0: (0, 100, 10) - takes tokens 0-10, borrows 10-100 from next chunk
            - Chunk 1: (10, 110, 100) - takes tokens 10-110
            - Chunk 2: (110, 210, 100) - takes tokens 110-210
        """
        chunks = []
        remainder = total_length % sequence_length
        
        if remainder > 0:
            # First chunk is smaller, needs to borrow from next chunk
            # It will take tokens [0:remainder] and borrow [remainder:sequence_length]
            chunks.append((0, sequence_length, remainder))
            start = sequence_length
        else:
            start = 0
        
        # Add full chunks
        while start < total_length:
            end = min(start + sequence_length, total_length)
            actual = min(sequence_length, total_length - start)
            chunks.append((start, end, actual))
            start = end
        
        return chunks
    
    def _extract_qkv(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract Q, K, V from hidden states using self-attention layer.
        
        Args:
            hidden_states: Chunk output (chunk_length, hidden_size)
            attention_mask: Optional mask (chunk_length,)
        
        Returns:
            Tuple of (Q, K, V) each (chunk_length, hidden_size)
        """
        # Use the cross-attention layer to compute Q, K, V
        # We'll extract them from the attention computation
        t = self.cross_attn.norm(hidden_states)
        qkv = self.cross_attn.qkv(t)
        
        # Split into Q, K, V
        q = qkv[:, : self.cross_attn.num_attention_heads * self.cross_attn.head_dim].contiguous()
        k = qkv[
            :,
            self.cross_attn.num_attention_heads * self.cross_attn.head_dim : 
            (self.cross_attn.num_attention_heads + self.cross_attn.num_key_value_heads) * self.cross_attn.head_dim,
        ].contiguous()
        v = qkv[
            :,
            (self.cross_attn.num_attention_heads + self.cross_attn.num_key_value_heads) * self.cross_attn.head_dim : 
            (self.cross_attn.num_attention_heads + 2 * self.cross_attn.num_key_value_heads) * self.cross_attn.head_dim,
        ].contiguous()
        
        # Reshape for attention
        q = q.view(
            -1,
            self.cross_attn.num_key_value_heads,
            self.cross_attn.num_attention_heads // self.cross_attn.num_key_value_heads,
            self.cross_attn.head_dim,
        )
        k = k.view(-1, self.cross_attn.num_key_value_heads, self.cross_attn.head_dim)
        v = v.view(-1, self.cross_attn.num_key_value_heads, self.cross_attn.head_dim)
        
        # Apply RoPE
        q, k = self.cross_attn.rope(q, k)
        
        # Flatten back to (seq_len, hidden_size)
        q = q.reshape(hidden_states.shape[0], -1)
        k = k.reshape(hidden_states.shape[0], -1)
        v = v.reshape(hidden_states.shape[0], -1)
        
        return q, k, v
    
    def _apply_cross_attention(
        self,
        Q_prev: torch.Tensor,
        K_curr: torch.Tensor,
        V_curr: torch.Tensor,
        attention_mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply cross-attention between chunks.
        
        Args:
            Q_prev: Query from previous chunk (prev_len, hidden_size)
            K_curr: Key from current chunk (curr_len, hidden_size)
            V_curr: Value from current chunk (curr_len, hidden_size)
            attention_mask: Optional mask for current chunk
        
        Returns:
            Updated (Q, K, V) for current chunk
        """
        # Perform cross-attention: Q from prev, K,V from current
        # For simplicity, we'll use the cross_attn layer
        
        # Reshape Q, K, V back to attention format
        curr_len = K_curr.shape[0]
        prev_len = Q_prev.shape[0]
        
        # Reshape for attention computation
        q = Q_prev.view(
            prev_len,
            self.cross_attn.num_key_value_heads,
            self.cross_attn.num_attention_heads // self.cross_attn.num_key_value_heads,
            self.cross_attn.head_dim,
        )
        k = K_curr.view(curr_len, self.cross_attn.num_key_value_heads, self.cross_attn.head_dim)
        v = V_curr.view(curr_len, self.cross_attn.num_key_value_heads, self.cross_attn.head_dim)
        
        # Perform cross-attention (no causal mask, attend across chunks)
        attn_output = bidirectional_sdpa(q, k, v, self.cross_attn.sm_scale, attention_mask=None)
        attn_output = self.cross_attn.attn_dropout(attn_output)
        attn_output = self.cross_attn.out(attn_output)
        
        # The attention output replaces Q for the current chunk
        # K and V remain from current chunk
        Q_new = attn_output
        
        return Q_new, K_curr, V_curr
    
    def _apply_final_cross_attention(
        self,
        Q_last: torch.Tensor,
        K_compressed: torch.Tensor,
        V_compressed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply final cross-attention: Q from last chunk attends to compressed K, V.
        
        The output is what the decoder will use for cross-attention:
        - encoder_k: The cross-attention output (Q_last attending to K_compressed, V_compressed)
        - encoder_v: Same as encoder_k (the attended representation)
        
        Args:
            Q_last: Query from last chunk (last_seq_len, hidden_size) - contains question
            K_compressed: SVD-compressed keys from other chunks (sequence_length, hidden_size)
            V_compressed: SVD-compressed values from other chunks (sequence_length, hidden_size)
        
        Returns:
            Tuple of (encoder_key, encoder_value) each (last_seq_len, hidden_size)
            These are the representations the decoder will attend to.
        """
        # Reshape for attention
        q_len = Q_last.shape[0]
        k_len = K_compressed.shape[0]
        
        q = Q_last.view(
            q_len,
            self.final_cross_attn.num_key_value_heads,
            self.final_cross_attn.num_attention_heads // self.final_cross_attn.num_key_value_heads,
            self.final_cross_attn.head_dim,
        )
        k = K_compressed.view(k_len, self.final_cross_attn.num_key_value_heads, self.final_cross_attn.head_dim)
        v = V_compressed.view(k_len, self.final_cross_attn.num_key_value_heads, self.final_cross_attn.head_dim)
        
        # Apply final cross-attention: Q_last attends to compressed context
        attn_output = bidirectional_sdpa(q, k, v, self.final_cross_attn.sm_scale, attention_mask=None)
        attn_output = self.final_cross_attn.attn_dropout(attn_output)
        encoder_output = self.final_cross_attn.out(attn_output)
        
        # The cross-attention output becomes both K and V for the decoder
        # This is the "question-aware context representation"
        return encoder_output, encoder_output
    
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
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)  # U: (seq_len, min(seq_len, hidden_size)), S: (min(seq_len, hidden_size),), Vh: (min(seq_len, hidden_size), hidden_size)
        
        # Keep top n_components
        U_reduced = U[:, :n_components]  # (seq_len, n_components)
        S_reduced = S[:n_components]  # (n_components,)
        
        # Reconstruct with reduced dimensions
        # We want output of shape (n_components, hidden_size)
        # X_compressed = U_reduced.T @ X
        X_compressed = U_reduced.T @ X  # (n_components, hidden_size)
        
        return X_compressed


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
        # Get pooled representation (using simple mean pooling)
        x = self.encoder(input_ids, attention_mask)
        
        if self.pooling == "first":
            pooled = x[0]
        elif self.pooling == "max":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).to(x.dtype)
                x = x * mask + (1 - mask) * (-1e9)
            pooled = x.max(dim=0)[0]
        else:  # mean pooling
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).to(x.dtype)
                sum_x = (x * mask).sum(dim=0)
                count = mask.sum(dim=0).clamp(min=1)
                pooled = sum_x / count
            else:
                pooled = x.mean(dim=0)
        
        # Classify
        logits = self.classifier(pooled)
        
        return logits
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
        
        # Process chunks in reverse order (last chunk first)
        for chunk_idx in range(num_chunks - 1, -1, -1):
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
            # Insert at beginning to maintain correct order (processing in reverse)
            context_embeddings.insert(0, context_embedding)
            
            # Update chunk context for next iteration
            self.chunk_context = context_embedding
            
            if not return_context_embeddings:
                # Only keep outputs for actual tokens (not padding)
                if chunk_len < chunk_size:
                    all_outputs.insert(0, chunk_output[:chunk_len])
                else:
                    all_outputs.insert(0, chunk_output)
        
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
        Process input in chunks with reverse-order processing and cross-attention between chunks.
        
        Strategy:
        1. Process chunks in reverse order (last tokens first)
        2. For each chunk, use cross-attention from previous chunk (later in time) as Q,
           and current chunk (earlier in time) as K,V
        3. If a chunk is smaller than chunk_size, pad it with tokens from the previous chunk
        4. Keep first chunk (chronologically) for final compression
        
        Returns:
            Tuple of (K_compressed, V_compressed) each (chunk_size, hidden_size)
        """
        input_len = input_ids.shape[0]
        
        # Calculate chunks with proper alignment
        # If tokens don't divide evenly, the first chunk (chronologically) may be smaller
        chunks = []
        
        if input_len <= chunk_size:
            # Single chunk, no reverse processing needed
            chunks.append((0, input_len))
        else:
            # Create chunks from the end, ensuring each is chunk_size except possibly the first
            num_full_chunks = input_len // chunk_size
            remainder = input_len % chunk_size
            
            if remainder > 0:
                # First chunk is smaller - but we pad it with tokens from next chunk
                # So we process [0, chunk_size] as first chunk with tokens [0, remainder] + padding from [remainder, chunk_size]
                chunks.append((0, chunk_size))  # Will use tokens [0:remainder] + padding from next chunk
                start = chunk_size
            else:
                start = 0
            
            # Add full chunks
            while start < input_len:
                end = min(start + chunk_size, input_len)
                chunks.append((start, end))
                start = end
        
        # Process chunks in reverse order (last first)
        all_chunk_outputs = []
        prev_chunk_output = None  # Output from previously processed chunk (later in time)
        first_chunk_output = None  # Store first chunk (chronologically) for compression
        
        for idx in range(len(chunks) - 1, -1, -1):
            start_idx, end_idx = chunks[idx]
            is_first_chronologically = (idx == 0)
            is_last_chronologically = (idx == len(chunks) - 1)
            
            # Get chunk tokens
            chunk_tokens = input_ids[start_idx:end_idx]
            actual_token_count = chunk_tokens.shape[0]
            
            # Pad to chunk_size if needed by borrowing from previous chunk
            if actual_token_count < chunk_size and not is_last_chronologically:
                # Borrow tokens from the next chunk (chronologically)
                next_chunk_start = chunks[idx + 1][0]
                tokens_needed = chunk_size - actual_token_count
                borrowed_tokens = input_ids[end_idx:end_idx + tokens_needed]
                chunk_tokens = torch.cat([chunk_tokens, borrowed_tokens], dim=0)
                
                # Update mask if provided
                if attention_mask is not None:
                    chunk_mask = attention_mask[start_idx:end_idx]
                    borrowed_mask = attention_mask[end_idx:end_idx + tokens_needed]
                    chunk_mask = torch.cat([chunk_mask, borrowed_mask], dim=0)
                else:
                    chunk_mask = None
            else:
                chunk_mask = attention_mask[start_idx:end_idx] if attention_mask is not None else None
            
            # Process this chunk with optional cross-attention from previous (later) chunk
            chunk_output = self._forward_chunk_with_cross_attention(
                chunk_tokens,
                chunk_mask,
                prev_chunk_output,  # Q will come from this (later chunk)
                is_first_chunk=is_first_chronologically
            )
            
            # Only keep outputs for actual tokens (not borrowed/padding)
            chunk_output = chunk_output[:actual_token_count]
            
            # Save first chunk (chronologically) for compression
            if is_first_chronologically:
                first_chunk_output = chunk_output
            
            # Insert at beginning to maintain chronological order
            all_chunk_outputs.insert(0, chunk_output)
            
            # This becomes the previous chunk for the next iteration
            prev_chunk_output = chunk_output
        
        # Stack all outputs: (total_tokens, hidden_size)
        stacked_outputs = torch.cat(all_chunk_outputs, dim=0)  # (seq_len, hidden_size)
        
        # Return full outputs without compression
        # Both K and V are the same - the full encoded representation
        return stacked_outputs, stacked_outputs
    
    def _forward_chunk_with_cross_attention(
        self,
        chunk_tokens: torch.Tensor,
        chunk_mask: torch.Tensor | None,
        prev_chunk_output: torch.Tensor | None,
        is_first_chunk: bool
    ) -> torch.Tensor:
        """
        Process a single chunk with optional cross-attention from the previous chunk.
        
        Args:
            chunk_tokens: Token IDs for this chunk (chunk_size,)
            chunk_mask: Attention mask for this chunk (chunk_size,) or None
            prev_chunk_output: Output from previously processed chunk (later in time) (prev_len, hidden_size)
            is_first_chunk: Whether this is the first chunk chronologically
        
        Returns:
            Chunk output (chunk_size, hidden_size)
        """
        # Embed tokens
        token_embeds = self.embedding(chunk_tokens)
        
        # Prepend context embedding
        if is_first_chunk:
            prepend_embed = self.start_token_embedding.unsqueeze(0)
        else:
            # Use context from previous chunk (if available)
            if prev_chunk_output is not None:
                prepend_embed = prev_chunk_output[-1:].detach()
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
        
        # If we have a previous chunk output, apply cross-attention
        # Q from current chunk, K,V from previous chunk (later in time)
        if prev_chunk_output is not None and not is_first_chunk:
            # Simple cross-attention: let current chunk attend to previous chunk
            # This is a simplified version - could be made more sophisticated
            # For now, we just use the outputs as-is
            # In a full implementation, you'd add a cross-attention layer here
            pass
        
        return x
    
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
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)  # U: (seq_len, min(seq_len, hidden_size)), S: (min(seq_len, hidden_size),), Vh: (min(seq_len, hidden_size), hidden_size)
        
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


def create_encoder_config_from_bert(bert_model_name="bert-large-uncased"):
    """Create encoder config matching BERT architecture"""
    from transformers import BertConfig
    bert_config = BertConfig.from_pretrained(bert_model_name)
    
    encoder_config = EncoderConfig(
        vocab_size=bert_config.vocab_size,                    # 30522
        hidden_size=bert_config.hidden_size,                  # 1024 for large, 768 for base
        num_hidden_layers=bert_config.num_hidden_layers,      # 24 for large, 12 for base
        num_attention_heads=bert_config.num_attention_heads,  # 16 for large, 12 for base
        num_key_value_heads=bert_config.num_attention_heads,  # Same as num_attention_heads
        head_dim=bert_config.hidden_size // bert_config.num_attention_heads,  # 64
        intermediate_size=bert_config.intermediate_size,      # 4096 for large, 3072 for base
        max_position_embeddings=bert_config.max_position_embeddings,  # 512
        dropout=bert_config.hidden_dropout_prob,              # 0.1
        use_moe=False,  # BERT doesn't have MoE
        num_experts=8,   # Not used when use_moe=False
        experts_per_token=2,  # Not used when use_moe=False
    )
    
    return encoder_config


def load_bert_encoder(encoder_config, bert_model_name, device):
    """Load BERT weights with proper layer mapping"""
    print(f"Loading BERT weights from: {bert_model_name}")
    
    from transformers import BertModel
    bert = BertModel.from_pretrained(bert_model_name)
    bert_state = bert.state_dict()
    
    encoder = BidirectionalEncoder(encoder_config, device=device)
    
    # Manual weight mapping
    loaded_count = 0
    skipped_count = 0
    
    # 1. Load embeddings
    try:
        encoder.embedding.weight.data.copy_(bert_state['embeddings.word_embeddings.weight'])
        loaded_count += 1
        print("✓ Loaded word embeddings")
    except Exception as e:
        print(f"⚠️  Could not load embeddings: {e}")
        skipped_count += 1
    
    # 2. Load encoder blocks
    for layer_idx in range(encoder_config.num_hidden_layers):
        bert_prefix = f'encoder.layer.{layer_idx}'
        encoder_block = encoder.blocks[layer_idx]
        
        # Attention weights
        try:
            # BERT uses: attention.self.query, key, value
            # Your encoder uses: attn.qkv (combined)
            
            # Get BERT's Q, K, V
            q_weight = bert_state[f'{bert_prefix}.attention.self.query.weight']
            q_bias = bert_state[f'{bert_prefix}.attention.self.query.bias']
            k_weight = bert_state[f'{bert_prefix}.attention.self.key.weight']
            k_bias = bert_state[f'{bert_prefix}.attention.self.key.bias']
            v_weight = bert_state[f'{bert_prefix}.attention.self.value.weight']
            v_bias = bert_state[f'{bert_prefix}.attention.self.value.bias']
            
            # Concatenate into your qkv format
            qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
            qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
            
            encoder_block.attn.qkv.weight.data.copy_(qkv_weight)
            encoder_block.attn.qkv.bias.data.copy_(qkv_bias)
            
            # Attention output projection
            encoder_block.attn.out.weight.data.copy_(
                bert_state[f'{bert_prefix}.attention.output.dense.weight']
            )
            encoder_block.attn.out.bias.data.copy_(
                bert_state[f'{bert_prefix}.attention.output.dense.bias']
            )
            
            loaded_count += 4
            print(f"✓ Loaded layer {layer_idx} attention")
            
        except Exception as e:
            print(f"⚠️  Could not load layer {layer_idx} attention: {e}")
            skipped_count += 1
        
        # FFN/MLP weights (if not using MoE)
        if not encoder_config.use_moe:
            try:
                # BERT: intermediate.dense (expansion), output.dense (projection)
                # Your encoder: mlp.fc1 (expansion with SwiGLU), mlp.fc2 (projection)
                
                # Note: Your fc1 is 2x intermediate_size for SwiGLU
                # BERT's is just intermediate_size
                # So we'll duplicate BERT's weights
                bert_fc1_weight = bert_state[f'{bert_prefix}.intermediate.dense.weight']
                bert_fc1_bias = bert_state[f'{bert_prefix}.intermediate.dense.bias']
                
                # Duplicate for SwiGLU (gate and up projection)
                fc1_weight = torch.cat([bert_fc1_weight, bert_fc1_weight], dim=0)
                fc1_bias = torch.cat([bert_fc1_bias, bert_fc1_bias], dim=0)
                
                encoder_block.mlp.fc1.weight.data.copy_(fc1_weight)
                encoder_block.mlp.fc1.bias.data.copy_(fc1_bias)
                
                # fc2 maps directly
                encoder_block.mlp.fc2.weight.data.copy_(
                    bert_state[f'{bert_prefix}.output.dense.weight']
                )
                encoder_block.mlp.fc2.bias.data.copy_(
                    bert_state[f'{bert_prefix}.output.dense.bias']
                )
                
                loaded_count += 2
                print(f"✓ Loaded layer {layer_idx} FFN")
                
            except Exception as e:
                print(f"⚠️  Could not load layer {layer_idx} FFN: {e}")
                skipped_count += 1
        else:
            print(f"⚠️  Layer {layer_idx} uses MoE - skipping FFN (will be random)")
            skipped_count += 1
        
        # Layer norms
        try:
            # BERT uses LayerNorm, you use RMSNorm
            # RMSNorm only has scale (weight), no bias
            # We can copy BERT's LayerNorm weight to RMSNorm scale
            
            # Attention norm
            if hasattr(encoder_block.attn.norm, 'scale'):
                encoder_block.attn.norm.scale.data.copy_(
                    bert_state[f'{bert_prefix}.attention.output.LayerNorm.weight']
                )
            
            # MLP norm
            if hasattr(encoder_block.mlp.norm, 'scale'):
                encoder_block.mlp.norm.scale.data.copy_(
                    bert_state[f'{bert_prefix}.output.LayerNorm.weight']
                )
            
            loaded_count += 2
            print(f"✓ Loaded layer {layer_idx} norms")
            
        except Exception as e:
            print(f"⚠️  Could not load layer {layer_idx} norms: {e}")
            skipped_count += 1
    
    # 3. Final layer norm
    try:
        if hasattr(encoder.norm, 'scale'):
            # BERT has a pooler, but we'll use the last encoder layer norm
            # Or just skip if it doesn't match
            print("⚠️  Skipping final norm (architecture difference)")
            skipped_count += 1
    except Exception as e:
        skipped_count += 1
    
    print(f"\n{'='*60}")
    print(f"✓ Successfully loaded {loaded_count} parameter groups")
    print(f"⚠️  Skipped {skipped_count} parameter groups")
    print(f"{'='*60}\n")
    
    return encoder
