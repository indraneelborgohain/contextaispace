"""
LSI Cross-Attention Block

This module implements cross-attention using Latent Semantic Indexing (LSI) to derive
Q, K, V matrices from stacked context vectors across transformer layers.
"""

import math
import torch
import torch.nn as nn
import numpy as np

from .lsi import LatentSemanticIndexing, LSIConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(
        self, num_features: int, eps: float = 1e-05, device: torch.device | None = None
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.scale = nn.Parameter(
            torch.ones(num_features, device=device, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.num_features
        t, dtype = x.float(), x.dtype
        t = t * torch.rsqrt(torch.mean(t**2, dim=-1, keepdim=True) + self.eps)
        return (t * self.scale).to(dtype)


class LSICrossAttentionBlock(nn.Module):
    """
    Cross-attention block using LSI-derived Q, K, V matrices.
    
    Takes stacked context vectors from all transformer layers, applies LSI
    to extract semantic structure, and performs cross-attention.
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
        
        # LSI configuration
        self.lsi_config = LSIConfig(
            n_components=lsi_components,
            normalize=True,
            use_idf=False
        )
        self.lsi = LatentSemanticIndexing(self.lsi_config)
        
        # Projection layers to/from LSI space
        self.to_lsi = nn.Linear(
            hidden_size, lsi_components, 
            device=device, dtype=torch.bfloat16
        )
        self.from_lsi = nn.Linear(
            lsi_components, hidden_size,
            device=device, dtype=torch.bfloat16
        )
        
        # Output projection
        self.out_proj = nn.Linear(
            hidden_size, hidden_size,
            device=device, dtype=torch.bfloat16
        )
        
        self.norm = RMSNorm(hidden_size, device=device)
        self.scale = 1.0 / math.sqrt(lsi_components)
    
    def forward(self, x: torch.Tensor, context_stack: torch.Tensor) -> torch.Tensor:
        """
        Apply LSI-based cross-attention.
        
        Args:
            x: Current hidden states (seq_len, hidden_size)
            context_stack: Stacked context vectors from all layers (num_layers, seq_len, hidden_size)
        
        Returns:
            Cross-attended output (seq_len, hidden_size)
        """
        seq_len = x.shape[0]
        num_layers = context_stack.shape[0]
        
        # Reshape context_stack to (num_layers * seq_len, hidden_size)
        context_flat = context_stack.reshape(-1, self.hidden_size)
        
        # Convert to numpy for LSI processing
        context_np = context_flat.detach().cpu().float().numpy()
        
        # Apply LSI to get Q, K, V matrices
        Q_np, K_np, V_np = self.lsi.fit_context_vectors(context_np, n_components=self.lsi_components)
        
        # Convert back to torch tensors
        Q = torch.from_numpy(Q_np).to(x.device).to(torch.bfloat16)
        K = torch.from_numpy(K_np).to(x.device).to(torch.bfloat16)
        V = torch.from_numpy(V_np).to(x.device).to(torch.bfloat16)
        
        # Q, K, V have shape (num_layers * seq_len, actual_components)
        # where actual_components may be less than lsi_components due to SVD constraints
        actual_components = Q.shape[1]
        
        # Reshape Q, K, V back to (num_layers, seq_len, actual_components)
        Q = Q.reshape(num_layers, seq_len, actual_components)
        K = K.reshape(num_layers, seq_len, actual_components)
        V = V.reshape(num_layers, seq_len, actual_components)
        
        # Aggregate across layers (mean pooling)
        Q_agg = Q.mean(dim=0)  # (seq_len, actual_components)
        K_agg = K.mean(dim=0)  # (seq_len, actual_components)
        V_agg = V.mean(dim=0)  # (seq_len, actual_components)
        
        # Compute attention scores: Q @ K^T with dynamic scaling
        scale = 1.0 / math.sqrt(actual_components)
        attn_scores = torch.matmul(Q_agg, K_agg.transpose(-2, -1)) * scale
        
        # Apply softmax
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # Apply attention to values: attn_weights @ V
        attn_output = torch.matmul(attn_weights, V_agg)  # (seq_len, actual_components)
        
        # Project back to hidden_size
        # Handle variable dimensionality from SVD
        if actual_components != self.lsi_components:
            # Create a temporary projection layer for this actual_components size
            temp_proj = nn.Linear(
                actual_components, self.hidden_size,
                device=x.device, dtype=torch.bfloat16
            )
            attn_output = temp_proj(attn_output)
        else:
            attn_output = self.from_lsi(attn_output)
        
        # Residual connection and normalization
        x = x + self.out_proj(attn_output)
        x = self.norm(x)
        
        return x
