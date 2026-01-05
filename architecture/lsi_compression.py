"""
LSI Compression Layer with SVD + Learned Projections

This module provides compression functionality using SVD followed by learned
projections to Q, K, V spaces. Used by both encoder and decoder for compressing
sequence representations.
"""

import torch
import torch.nn as nn
from .attention_components import RMSNorm


class LSICompressionLayer(nn.Module):
    """
    LSI Compression Layer with SVD + Learned Projections.
    
    If seq_len > target_length:
        1. SVD compress: (seq_len, hidden_size) → (target_length, hidden_size)
        2. Project to K, V spaces using learned matrices
    Else if seq_len < target_length:
        1. Cyclically repeat to reach target_length
        2. Project to K, V spaces
    Else:
        Just project to K, V spaces
    
    Returns separate K and V compressed representations.
    """
    def __init__(
        self,
        hidden_size: int,
        target_length: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.target_length = target_length
        
        # Learned projection matrices (hidden_size, hidden_size)
        self.key_proj = nn.Linear(
            hidden_size,
            hidden_size,
            device=device,
            dtype=torch.bfloat16
        )
        self.value_proj = nn.Linear(
            hidden_size,
            hidden_size,
            device=device,
            dtype=torch.bfloat16
        )
        
        self.norm = RMSNorm(hidden_size, device=device)
    
    def forward(self, input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compress input using SVD + learned projections.
        
        Args:
            input_tensor: Input tensor (seq_len, hidden_size)
        
        Returns:
            Tuple of (K_compressed, V_compressed) each (target_length, hidden_size)
        """
        seq_len, hidden_size = input_tensor.shape
        
        # Normalize input
        input_tensor = self.norm(input_tensor)
        
        # Step 1: SVD compression or cyclic expansion
        if seq_len > self.target_length:
            # Perform SVD: input_tensor = U @ S @ V^T
            U, S, Vt = torch.svd(input_tensor)
            
            # Keep top target_length components
            U_reduced = U[:, :self.target_length]  # (seq_len, target_length)
            
            # Compress: X_compressed = U_reduced.T @ input_tensor
            compressed = U_reduced.T @ input_tensor  # (target_length, hidden_size)
        else:
            # No compression needed, but expand to target_length if shorter
            if seq_len < self.target_length:
                # Repeat content cyclically to reach target_length
                # E.g., 50 tokens → 128: repeat 50 twice (100) + first 28 (128)
                num_repeats = self.target_length // seq_len
                remainder = self.target_length % seq_len
                
                # Repeat full sequence
                repeated = input_tensor.repeat(num_repeats, 1)  # (num_repeats * seq_len, hidden_size)
                
                # Add partial repeat for remainder
                if remainder > 0:
                    partial = input_tensor[:remainder]  # (remainder, hidden_size)
                    compressed = torch.cat([repeated, partial], dim=0)  # (target_length, hidden_size)
                else:
                    compressed = repeated
            else:
                # Already exact target length
                compressed = input_tensor
        
        # Step 2: Project to K and V spaces using learned matrices
        K_compressed = self.key_proj(compressed)    # (target_length, hidden_size)
        V_compressed = self.value_proj(compressed)  # (target_length, hidden_size)
        
        return K_compressed, V_compressed
