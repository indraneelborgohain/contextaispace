"""
LSI Cross-Attention Block

This module implements cross-attention using SVD compression on stacked chunk outputs,
followed by learned projections to Q, K, V spaces (similar to encoder compression).
"""

import math
import torch
import torch.nn as nn

from .attention_components import RMSNorm
from .lsi_compression import LSICompressionLayer


class LSICrossAttentionBlock(nn.Module):
    """
    Cross-attention block using SVD + learned Q, K, V projections.
    
    Takes stacked chunk outputs, applies SVD compression + learned projections
    to get Q, K, V, then performs cross-attention.
    """
    def __init__(
        self,
        hidden_size: int,
        lsi_components: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.lsi_components = lsi_components
        
        # LSI compression layer for Q, K, V
        self.q_compression = LSICompressionLayer(
            hidden_size=hidden_size,
            target_length=lsi_components,
            device=device
        )
        # Note: LSICompressionLayer returns K and V, but we'll use it for Q too
        # For Q, we'll just use the K projection output
        
        # Additional projection for Query (since we need separate Q)
        self.q_proj = nn.Linear(
            hidden_size, hidden_size,
            device=device, dtype=torch.bfloat16
        )
        
        # Output projection
        self.out_proj = nn.Linear(
            hidden_size, hidden_size,
            device=device, dtype=torch.bfloat16
        )
        
        self.norm = RMSNorm(hidden_size, device=device)
        self.sm_scale = 1.0 / math.sqrt(hidden_size)
    
    def forward(self, x: torch.Tensor, chunk_outputs: torch.Tensor) -> torch.Tensor:
        """
        Apply LSI-based cross-attention using stacked chunk outputs.
        
        Args:
            x: Current hidden states (seq_len, hidden_size)
            chunk_outputs: Stacked outputs from all chunks (total_chunk_tokens, hidden_size)
        
        Returns:
            Cross-attended output (seq_len, hidden_size)
        """
        residual = x
        
        # Apply SVD compression + learned projections to get K, V
        K_compressed, V_compressed = self.q_compression(chunk_outputs)  # Each (lsi_components, hidden_size)
        
        # Generate Q from current hidden states
        # First compress if needed
        seq_len = x.shape[0]
        if seq_len > self.lsi_components:
            # Compress x using SVD
            U, S, Vt = torch.svd(x)
            U_reduced = U[:, :self.lsi_components]
            x_compressed = U_reduced.T @ x  # (lsi_components, hidden_size)
        elif seq_len < self.lsi_components:
            # Expand cyclically
            num_repeats = self.lsi_components // seq_len
            remainder = self.lsi_components % seq_len
            repeated = x.repeat(num_repeats, 1)
            if remainder > 0:
                partial = x[:remainder]
                x_compressed = torch.cat([repeated, partial], dim=0)
            else:
                x_compressed = repeated
        else:
            x_compressed = x
        
        Q_compressed = self.q_proj(x_compressed)  # (lsi_components, hidden_size)
        
        # Compute attention scores: Q @ K^T
        attn_scores = torch.matmul(Q_compressed, K_compressed.transpose(-2, -1))  # (lsi_components, lsi_components)
        attn_scores = attn_scores * self.sm_scale
        
        # Apply softmax
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V_compressed)  # (lsi_components, hidden_size)
        
        # Project output
        attn_output = self.out_proj(attn_output)  # (lsi_components, hidden_size)
        
        # Resize back to original seq_len if needed
        if attn_output.shape[0] != seq_len:
            if seq_len > self.lsi_components:
                # Expand back
                # Use learned upsampling via linear interpolation or simple repeat
                attn_output = attn_output.repeat_interleave(
                    (seq_len + self.lsi_components - 1) // self.lsi_components, 
                    dim=0
                )[:seq_len]
            else:
                # Take first seq_len
                attn_output = attn_output[:seq_len]
        
        # Residual connection and normalization
        x = residual + attn_output
        x = self.norm(x)
        
        return x
