import math
import torch


class RMSNorm(torch.nn.Module):
    def __init__(
        self, num_features: int, eps: float = 1e-05, device: torch.device | None = None
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.scale = torch.nn.Parameter(
            torch.ones(num_features, device=device, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.num_features
        t, dtype = x.float(), x.dtype
        t = t * torch.rsqrt(torch.mean(t**2, dim=-1, keepdim=True) + self.eps)
        return (t * self.scale).to(dtype)


class CrossAttentionBlock(torch.nn.Module):
    """Cross-attention from decoder to encoder."""
    def __init__(
        self,
        decoder_hidden_size: int,
        encoder_hidden_size: int,
        head_dim: int = 64,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 8,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        
        # Norm for decoder input
        self.norm_q = RMSNorm(decoder_hidden_size, device=device)
        
        # Query projection from decoder
        self.q_proj = torch.nn.Linear(
            decoder_hidden_size,
            num_attention_heads * head_dim,
            bias=False,
            device=device,
            dtype=torch.bfloat16,
        )
        
        # Key/Value projections from encoder
        kv_dim = num_key_value_heads * head_dim
        self.k_proj = torch.nn.Linear(
            encoder_hidden_size,
            kv_dim,
            bias=False,
            device=device,
            dtype=torch.bfloat16,
        )
        self.v_proj = torch.nn.Linear(
            encoder_hidden_size,
            kv_dim,
            bias=False,
            device=device,
            dtype=torch.bfloat16,
        )
        
        # Output projection
        self.out = torch.nn.Linear(
            num_attention_heads * head_dim,
            decoder_hidden_size,
            bias=False,
            device=device,
            dtype=torch.bfloat16,
        )
        
        # Gating parameter for gradual integration
        self.angle = torch.nn.Parameter(
            torch.tensor(0.1, device=device, dtype=torch.bfloat16)
        )
        
        self.sm_scale = 1 / math.sqrt(head_dim)

    def forward(
        self, 
        x: torch.Tensor, 
        encoder_output: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: decoder hidden states [seq_len, hidden_size] or [batch, seq_len, hidden_size]
            encoder_output: encoder hidden states [ctx_len, enc_hidden] or [batch, ctx_len, enc_hidden]
        """
        residual = x
        
        # Handle both batched and unbatched inputs
        is_batched = x.dim() == 3
        if not is_batched:
            x = x.unsqueeze(0)
            encoder_output = encoder_output.unsqueeze(0)
        
        batch_size, seq_len, _ = x.shape
        _, ctx_len, _ = encoder_output.shape
        
        # Normalize decoder input
        q = self.norm_q(x)
        
        # Project Q, K, V
        q = self.q_proj(q)  # [batch, seq_len, num_heads * head_dim]
        k = self.k_proj(encoder_output)  # [batch, ctx_len, num_kv_heads * head_dim]
        v = self.v_proj(encoder_output)  # [batch, ctx_len, num_kv_heads * head_dim]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        k = k.view(batch_size, ctx_len, self.num_key_value_heads, self.head_dim)
        v = v.view(batch_size, ctx_len, self.num_key_value_heads, self.head_dim)
        
        # Transpose for attention: [batch, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Expand KV heads to match Q heads (GQA)
        if self.num_key_value_heads < self.num_attention_heads:
            k = k.repeat_interleave(
                self.num_attention_heads // self.num_key_value_heads, dim=1
            )
            v = v.repeat_interleave(
                self.num_attention_heads // self.num_key_value_heads, dim=1
            )
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.sm_scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back: [batch, seq_len, num_heads * head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        
        # Output projection
        output = self.out(attn_output)
        
        # Gated residual connection
        gate = torch.tanh(self.angle)
        output = residual + gate * output
        
        # Remove batch dimension if input was unbatched
        if not is_batched:
            output = output.squeeze(0)
        
        return output
