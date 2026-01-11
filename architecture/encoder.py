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
    
    Args:
        Q: Query tensor (q_len, n_heads, q_mult, d_head)
        K: Key tensor (k_len, n_heads, d_head)
        V: Value tensor (k_len, n_heads, d_head)
        sm_scale: Scaling factor for attention scores
        attention_mask: Optional attention mask (q_len, k_len), True = keep, False = mask
    
    Returns:
        Attention output (q_len, n_heads * q_mult * d_head)
    """
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
        sequence_length: int | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input tokens with reverse-order chunk processing.
        
        Args:
            input_ids: Input token IDs (total_sequence_length,)
            attention_mask: Optional mask (total_sequence_length,) where 1 = real token, 0 = padding
            return_encoder_kv: If True, returns (encoder_key, encoder_value) for decoder cross-attention
            sequence_length: Maximum sequence length per chunk. If None, uses max_position_embeddings
        
        Returns:
            If return_encoder_kv=False: Encoded representations (total_sequence_length, hidden_size)
            If return_encoder_kv=True: Tuple of (encoder_key, encoder_value) each (sequence_length, hidden_size)
        """
        if sequence_length is None:
            sequence_length = self.config.max_position_embeddings
        
        total_length = input_ids.shape[0]
        
        # Single chunk case - no chunking needed
        if total_length <= sequence_length:
            x = self._process_single_chunk(input_ids, attention_mask)
            if return_encoder_kv:
                # For single chunk, just return the output as both K and V
                return x, x
            return x
        
        # Multi-chunk case - process in reverse order
        if return_encoder_kv:
            return self._forward_with_chunking(input_ids, attention_mask, sequence_length)
        else:
            # Just return the encoded representation without K,V extraction
            encoder_k, encoder_v = self._forward_with_chunking(input_ids, attention_mask, sequence_length)
            # Concatenate or return first chunk
            return encoder_k
    
    def _process_single_chunk(
        self,
        chunk_tokens: torch.Tensor,
        chunk_mask: torch.Tensor | None
    ) -> torch.Tensor:
        """
        Process a single chunk through transformer blocks.
        
        Args:
            chunk_tokens: Token IDs (chunk_length,)
            chunk_mask: Attention mask (chunk_length,) or None
        
        Returns:
            Encoded output (chunk_length, hidden_size)
        """
        # Embed tokens
        x = self.embedding(chunk_tokens)
        x = self.dropout(x)
        
        # Create attention mask if provided
        if chunk_mask is not None:
            seq_len = chunk_mask.shape[0]
            attn_mask_2d = chunk_mask.unsqueeze(0) & chunk_mask.unsqueeze(1)
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
        sequence_length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Process input in reverse-order chunks with cross-attention.
        
        Strategy:
        1. Split into chunks, ensuring each is exactly sequence_length
        2. Process chunks in reverse order (last first)
        3. Apply cross-attention between consecutive chunks
        4. Final cross-attention from first processed to last processed chunk
        
        Args:
            input_ids: Input token IDs (total_sequence_length,)
            attention_mask: Optional mask (total_sequence_length,)
            sequence_length: Target length for each chunk
        
        Returns:
            Tuple of (encoder_key, encoder_value) each (sequence_length, hidden_size)
        """
        total_length = input_ids.shape[0]
        
        # Create chunks with proper alignment
        chunks = self._create_chunks(total_length, sequence_length)
        
        # Storage for chunk outputs
        Q_prev = None  # Query from previously processed chunk (later in time)
        Q_first_processed = None  # Query from first chunk we process (last chronologically)
        K_final = None  # Key from final chunk (first chronologically)
        V_final = None  # Value from final chunk (first chronologically)
        
        # Process chunks in reverse order
        for idx in range(len(chunks) - 1, -1, -1):
            start_idx, end_idx, actual_length = chunks[idx]
            is_first_processed = (idx == len(chunks) - 1)  # Last chronologically, first processed
            is_last_processed = (idx == 0)  # First chronologically, last processed
            
            # Extract chunk tokens (with borrowing if needed)
            chunk_tokens = input_ids[start_idx:end_idx]
            chunk_mask = attention_mask[start_idx:end_idx] if attention_mask is not None else None
            
            # Process chunk through transformer blocks
            chunk_output = self._process_single_chunk(chunk_tokens, chunk_mask)
            
            # Extract Q, K, V from chunk output using self-attention
            Q_chunk, K_chunk, V_chunk = self._extract_qkv(chunk_output, chunk_mask)
            
            # Apply cross-attention if not the first chunk being processed
            if Q_prev is not None:
                # Cross-attention: Q from previous chunk, K,V from current chunk
                Q_chunk, K_chunk, V_chunk = self._apply_cross_attention(
                    Q_prev, K_chunk, V_chunk, chunk_mask
                )
            
            # Save first processed chunk's Q for final cross-attention
            if is_first_processed:
                Q_first_processed = Q_chunk
            
            # Save last processed chunk's K,V (first chronologically)
            if is_last_processed:
                K_final = K_chunk[:actual_length]  # Only keep actual tokens
                V_final = V_chunk[:actual_length]
            
            # Update Q_prev for next iteration
            Q_prev = Q_chunk
        
        # Final cross-attention: Q from first processed, K,V from last processed
        encoder_key, encoder_value = self._apply_final_cross_attention(
            Q_first_processed, K_final, V_final
        )
        
        return encoder_key, encoder_value
    
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
        Q_first: torch.Tensor,
        K_last: torch.Tensor,
        V_last: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply final cross-attention from first processed to last processed chunk.
        
        Args:
            Q_first: Query from first processed chunk (seq_len, hidden_size)
            K_last: Key from last processed chunk (seq_len, hidden_size)
            V_last: Value from last processed chunk (seq_len, hidden_size)
        
        Returns:
            Tuple of (encoder_key, encoder_value) each (seq_len, hidden_size)
        """
        # Reshape for attention
        seq_len = Q_first.shape[0]
        k_len = K_last.shape[0]
        
        q = Q_first.view(
            seq_len,
            self.final_cross_attn.num_key_value_heads,
            self.final_cross_attn.num_attention_heads // self.final_cross_attn.num_key_value_heads,
            self.final_cross_attn.head_dim,
        )
        k = K_last.view(k_len, self.final_cross_attn.num_key_value_heads, self.final_cross_attn.head_dim)
        v = V_last.view(k_len, self.final_cross_attn.num_key_value_heads, self.final_cross_attn.head_dim)
        
        # Apply final cross-attention
        attn_output = bidirectional_sdpa(q, k, v, self.final_cross_attn.sm_scale, attention_mask=None)
        attn_output = self.final_cross_attn.attn_dropout(attn_output)
        encoder_output = self.final_cross_attn.out(attn_output)
        
        # Return both as encoder K and V
        return encoder_output, encoder_output


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
