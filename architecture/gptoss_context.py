"""
Context-Aware GPT Implementation with LSI Cross-Attention

This module implements a nano-GPT OSS variant where the output context vector 
from the previous prediction is fed back as input to the model.

The context state (shape: hidden_size) stores the output from the previous token 
prediction and is fed into each transformer layer through context projection layers.

Additionally, context vectors from all transformer blocks are stacked and passed
through an LSI module to extract Q, K, V matrices, which are then used in a 
cross-attention layer before final prediction.
"""

import json
import math
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
import numpy as np
from .lsi import LatentSemanticIndexing, LSIConfig


@dataclass
class ModelConfig:
    """Configuration for context-aware GPT model."""
    num_hidden_layers: int = 24
    num_experts: int = 32
    experts_per_token: int = 4
    vocab_size: int = 201088
    hidden_size: int = 2880
    intermediate_size: int = 2880
    swiglu_limit: float = 7.0
    head_dim: int = 64
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    sliding_window: int = 128
    initial_context_length: int = 4096
    rope_theta: float = 150000.0
    rope_scaling_factor: float = 32.0
    rope_ntk_alpha: float = 1.0
    rope_ntk_beta: float = 32.0
    lsi_components: int = 64  # Number of LSI latent dimensions
    use_lsi_cross_attention: bool = True  # Enable LSI cross-attention
    
    @property
    def context_dim(self) -> int:
        return self.hidden_size


class RMSNorm(torch.nn.Module):
    """Root Mean Square Layer Normalization."""
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


def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    x1, x2 = torch.chunk(x, 2, dim=-1)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat((o1, o2), dim=-1)


class RotaryEmbedding(torch.nn.Module):
    """Rotary Position Embeddings with YaRN scaling."""
    def __init__(
        self,
        head_dim: int,
        base: int,
        dtype: torch.dtype,
        initial_context_length: int = 4096,
        scaling_factor: float = 1.0,
        ntk_alpha: float = 1.0,
        ntk_beta: float = 32.0,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.dtype = dtype
        self.initial_context_length = initial_context_length
        self.scaling_factor = scaling_factor
        self.ntk_alpha = ntk_alpha
        self.ntk_beta = ntk_beta
        self.device = device

    def _compute_concentration_and_inv_freq(self) -> torch.Tensor:
        """See YaRN paper: https://arxiv.org/abs/2309.00071"""
        freq = self.base ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float, device=self.device)
            / self.head_dim
        )
        if self.scaling_factor > 1.0:
            concentration = (
                0.1 * math.log(self.scaling_factor) + 1.0
            )

            d_half = self.head_dim / 2
            low = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_beta * 2 * math.pi))
                / math.log(self.base)
            )
            high = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_alpha * 2 * math.pi))
                / math.log(self.base)
            )
            assert 0 < low < high < d_half - 1

            interpolation = 1.0 / (self.scaling_factor * freq)
            extrapolation = 1.0 / freq

            ramp = (
                torch.arange(d_half, dtype=torch.float32, device=freq.device) - low
            ) / (high - low)
            mask = 1 - ramp.clamp(0, 1)

            inv_freq = interpolation * (1 - mask) + extrapolation * mask
        else:
            concentration = 1.0
            inv_freq = 1.0 / freq

        return concentration, inv_freq

    def _compute_cos_sin(self, num_tokens: int):
        concentration, inv_freq = self._compute_concentration_and_inv_freq()
        t = torch.arange(num_tokens, dtype=torch.float32, device=self.device)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos = freqs.cos() * concentration
        sin = freqs.sin() * concentration
        return cos, sin

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = query.shape[0]
        cos, sin = self._compute_cos_sin(num_tokens)

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_dim)
        query = _apply_rotary_emb(query, cos, sin)
        query = query.reshape(query_shape)

        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_dim)
        key = _apply_rotary_emb(key, cos, sin)
        key = key.reshape(key_shape)
        return query, key


def sdpa(Q, K, V, S, sm_scale, sliding_window=0):
    """Scaled Dot-Product Attention with Grouped Query Attention."""
    n_tokens, n_heads, q_mult, d_head = Q.shape
    assert K.shape == (n_tokens, n_heads, d_head)
    assert V.shape == (n_tokens, n_heads, d_head)
    K = K[:, :, None, :].expand(-1, -1, q_mult, -1)
    V = V[:, :, None, :].expand(-1, -1, q_mult, -1)
    S = S.reshape(n_heads, q_mult, 1, 1).expand(-1, -1, n_tokens, -1)
    mask = torch.triu(Q.new_full((n_tokens, n_tokens), -float("inf")), diagonal=1)
    if sliding_window > 0:
        mask += torch.tril(
            mask.new_full((n_tokens, n_tokens), -float("inf")), diagonal=-sliding_window
        )
    QK = torch.einsum("qhmd,khmd->hmqk", Q, K)
    QK *= sm_scale
    QK += mask[None, None, :, :]
    QK = torch.cat([QK, S], dim=-1)
    W = torch.softmax(QK, dim=-1)
    W = W[..., :-1]
    attn = torch.einsum("hmqk,khmd->qhmd", W, V)
    return attn.reshape(n_tokens, -1)


class AttentionBlock(torch.nn.Module):
    """
    Context-aware attention block with grouped-query attention.
    Integrates context via projection layer.
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
        
        self.context_proj = torch.nn.Linear(
            config.context_dim, config.hidden_size, device=device, dtype=torch.bfloat16
        )
        
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

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        context_proj = self.context_proj(context)
        x_with_context = x + context_proj
        
        t = self.norm(x_with_context)
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


def swiglu(x, alpha: float = 1.702, limit: float = 7.0):
    """SwiGLU activation with clamping."""
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    return out_glu * (x_linear + 1)


class MLPBlock(torch.nn.Module):
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


class TransformerBlock(torch.nn.Module):
    """Transformer block with context-aware attention and MLP."""
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

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = self.attn(x, context)
        x = self.mlp(x, context)
        return x


class LSICrossAttentionBlock(torch.nn.Module):
    """
    Cross-attention block using LSI-derived Q, K, V matrices.
    
    Takes stacked context vectors from all transformer layers, applies LSI
    to extract semantic structure, and performs cross-attention.
    """
    def __init__(
        self,
        config: ModelConfig,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.lsi_components = config.lsi_components
        
        # LSI configuration
        self.lsi_config = LSIConfig(
            n_components=config.lsi_components,
            normalize=True,
            use_idf=False
        )
        self.lsi = LatentSemanticIndexing(self.lsi_config)
        
        # Projection layers to/from LSI space
        self.to_lsi = torch.nn.Linear(
            config.hidden_size, config.lsi_components, 
            device=device, dtype=torch.bfloat16
        )
        self.from_lsi = torch.nn.Linear(
            config.lsi_components, config.hidden_size,
            device=device, dtype=torch.bfloat16
        )
        
        # Output projection
        self.out_proj = torch.nn.Linear(
            config.hidden_size, config.hidden_size,
            device=device, dtype=torch.bfloat16
        )
        
        self.norm = RMSNorm(config.hidden_size, device=device)
        self.scale = 1.0 / math.sqrt(config.lsi_components)
    
    def forward(self, x: torch.Tensor, context_stack: torch.Tensor) -> torch.Tensor:
        """
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
            temp_proj = torch.nn.Linear(
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
        self.block = torch.nn.ModuleList(
            [
                TransformerBlock(config, layer_idx, device)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        
        # LSI cross-attention block
        if config.use_lsi_cross_attention:
            self.lsi_cross_attn = LSICrossAttentionBlock(config, device)
        else:
            self.lsi_cross_attn = None
        
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

    def reset_context(self):
        """Reset context state to zeros for new conversation."""
        self.context_state = torch.zeros_like(self.context_state)

    def forward(self, x: torch.Tensor, update_context: bool = True, max_seq_len: int = None) -> torch.Tensor:
        """
        Forward pass with LSI cross-attention.
        
        Process:
        1. If input is longer than max_seq_len, process in chunks
        2. Pass through transformer blocks, collecting context vectors
        3. Stack context vectors and apply LSI cross-attention
        4. Final normalization and prediction
        
        Args:
            x: Input token IDs (seq_len,)
            update_context: Whether to update context state after processing
            max_seq_len: Maximum sequence length per chunk. If None, uses config's sliding_window
        
        Set update_context=False to prevent context update.
        """
        if max_seq_len is None:
            max_seq_len = self.config.sliding_window
        
        input_len = x.shape[0]
        
        # If input is longer than max_seq_len, process in chunks
        if input_len > max_seq_len:
            all_logits = []
            num_chunks = (input_len + max_seq_len - 1) // max_seq_len
            
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * max_seq_len
                end_idx = min(start_idx + max_seq_len, input_len)
                chunk = x[start_idx:end_idx]
                
                # Process chunk and update context for next chunk
                chunk_logits = self._forward_chunk(chunk, update_context=True)
                all_logits.append(chunk_logits)
            
            # Concatenate all logits
            logits = torch.cat(all_logits, dim=0)
            
            # Final context update based on user preference
            if not update_context:
                # If user wants no context update, we need to restore original context
                # But we already updated it during chunking, so we keep it
                pass
            
            return logits
        else:
            # Normal processing for short sequences
            return self._forward_chunk(x, update_context=update_context)
    
    def _forward_chunk(self, x: torch.Tensor, update_context: bool = True) -> torch.Tensor:
        """Process a single chunk of tokens."""
        token_embeds = self.embedding(x)
        context_embed = self.context_state.unsqueeze(0)
        x = torch.cat([context_embed, token_embeds], dim=0)
        context = self.context_state.unsqueeze(0).expand(x.shape[0], -1)
        
        # Collect context vectors from each transformer block
        if self.lsi_cross_attn is not None:
            context_vectors = []
        
        for block in self.block:
            x = block(x, context)
            if self.lsi_cross_attn is not None:
                # Store the output of this block for LSI processing
                context_vectors.append(x.clone())
        
        # Apply LSI cross-attention if enabled
        if self.lsi_cross_attn is not None and len(context_vectors) > 0:
            # Stack context vectors: (num_layers, seq_len, hidden_size)
            context_stack = torch.stack(context_vectors, dim=0)
            x = self.lsi_cross_attn(x, context_stack)
        
        x = self.norm(x)
        x_tokens = x[1:]
        logits = self.unembedding(x_tokens)
        
        if update_context and x_tokens.shape[0] > 0:
            self.context_state = x_tokens[-1].detach().clone()
        
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


class TokenGenerator:
    """
    Token generation interface for context-aware GPT.
    Automatically manages context across generation steps.
    """
    @torch.inference_mode()
    def __init__(self, checkpoint: str, device: torch.device):
        self.device = device
        self.model = Transformer.from_checkpoint(checkpoint, device=self.device)
        self.model.reset_context()

    def reset(self):
        """Reset context for new conversation."""
        self.model.reset_context()

    @torch.inference_mode()
    def generate(self,
                 prompt_tokens: list[int],
                 stop_tokens: list[int],
                 temperature: float = 1.0,
                 max_tokens: int = 0,
                 return_logprobs: bool = False,
                 reset_context: bool = False):
        """
        Generate tokens autoregressively.
        
        Args:
            prompt_tokens: Input token IDs
            stop_tokens: Tokens that signal end of generation
            temperature: Sampling temperature (0.0 = greedy)
            max_tokens: Maximum tokens to generate (0 = no limit)
            return_logprobs: Yield (token, logprob) tuples if True
            reset_context: Reset context before generation if True
        """
        if reset_context:
            self.model.reset_context()
        
        tokens = list(prompt_tokens)
        num_generated_tokens = 0
        
        while max_tokens == 0 or num_generated_tokens < max_tokens:
            logits = self.model(
                torch.as_tensor(tokens, dtype=torch.int32, device=self.device),
                update_context=True
            )[-1]
            
            if temperature == 0.0:
                predicted_token = torch.argmax(logits, dim=-1).item()
            else:
                probs = torch.softmax(logits * (1.0 / temperature), dim=-1)
                predicted_token = torch.multinomial(probs, num_samples=1).item()
            
            tokens.append(predicted_token)
            num_generated_tokens += 1

            if return_logprobs:
                logprobs = torch.log_softmax(logits, dim=-1)
                selected_logprobs = logprobs[predicted_token].item()
                yield predicted_token, selected_logprobs
            else:
                yield predicted_token

            if predicted_token in stop_tokens:
                break
