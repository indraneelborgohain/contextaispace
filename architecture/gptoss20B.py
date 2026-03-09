import json
import math
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

from gpt_oss.torch.weights import Checkpoint


@dataclass
class ModelConfig:
    num_hidden_layers: int = 36
    num_experts: int = 128
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


def reposition_rope(
    k_cache: torch.Tensor,
    old_positions: torch.Tensor,
    new_positions: torch.Tensor,
    rope_module: "RotaryEmbedding",
) -> torch.Tensor:
    """Re-number positional encodings on cached K tensors.

    RoPE is reversible: apply_rotary(x, cos, -sin) undoes the original
    rotation, then we re-apply with the desired positions.

    Only K caches need this — V caches have no positional encoding.

    Args:
        k_cache: (seq_len, num_kv_heads, head_dim) — already-RoPE'd keys.
        old_positions: (seq_len,) int tensor of original position indices.
        new_positions: (seq_len,) int tensor of target position indices.
        rope_module: The RotaryEmbedding instance (provides YaRN freqs).

    Returns:
        Re-positioned K cache with same shape.
    """
    seq_len = k_cache.shape[0]
    concentration, inv_freq = rope_module._compute_concentration_and_inv_freq()

    def _cos_sin_for(positions: torch.Tensor):
        t = positions.float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        return freqs.cos() * concentration, freqs.sin() * concentration

    cos_old, sin_old = _cos_sin_for(old_positions)
    cos_new, sin_new = _cos_sin_for(new_positions)

    orig_shape = k_cache.shape
    k = k_cache.view(seq_len, -1, rope_module.head_dim)

    # Un-rotate: negate sin
    k = _apply_rotary_emb(k, cos_old, -sin_old)
    # Re-rotate at new positions
    k = _apply_rotary_emb(k, cos_new, sin_new)

    return k.reshape(orig_shape)


class RotaryEmbedding(torch.nn.Module):
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
            )  # YaRN concentration

            d_half = self.head_dim / 2
            # NTK by parts
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
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = query.shape[0]
        # Compute cos/sin for positions [start_pos, start_pos + num_tokens)
        cos, sin = self._compute_cos_sin(start_pos + num_tokens)
        cos = cos[start_pos:]
        sin = sin[start_pos:]

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_dim)
        query = _apply_rotary_emb(query, cos, sin)
        query = query.reshape(query_shape)

        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_dim)
        key = _apply_rotary_emb(key, cos, sin)
        key = key.reshape(key_shape)
        return query, key


def sdpa(Q, K, V, S, sm_scale, sliding_window=0, start_pos: int = 0):
    # sliding_window == 0 means no sliding window
    # start_pos: position of first Q token (for KV cache)
    q_len, n_heads, q_mult, d_head = Q.shape
    kv_len = K.shape[0]
    assert K.shape == (kv_len, n_heads, d_head)
    assert V.shape == (kv_len, n_heads, d_head)
    K = K[:, :, None, :].expand(-1, -1, q_mult, -1)
    V = V[:, :, None, :].expand(-1, -1, q_mult, -1)
    S = S.reshape(n_heads, q_mult, 1, 1).expand(-1, -1, q_len, -1)

    # Build causal mask: Q tokens can only attend to K tokens at positions <= their own
    # Q positions: [start_pos, start_pos + q_len)
    # K positions: [0, kv_len)
    q_positions = torch.arange(start_pos, start_pos + q_len, device=Q.device)
    k_positions = torch.arange(kv_len, device=Q.device)

    # Create mask using masked_fill to avoid 0 * -inf = NaN
    mask = torch.zeros((q_len, kv_len), device=Q.device, dtype=Q.dtype)
    # mask[i,j] = -inf if k_positions[j] > q_positions[i] (future token)
    causal_mask = k_positions[None, :] > q_positions[:, None]
    mask.masked_fill_(causal_mask, -float("inf"))

    if sliding_window > 0:
        # Also mask tokens outside sliding window
        # mask[i,j] = -inf if q_positions[i] - k_positions[j] >= sliding_window
        window_mask = q_positions[:, None] - k_positions[None, :] >= sliding_window
        mask.masked_fill_(window_mask, -float("inf"))

    QK = torch.einsum("qhmd,khmd->hmqk", Q, K)
    QK *= sm_scale
    QK += mask[None, None, :, :]
    QK = torch.cat([QK, S], dim=-1)
    W = torch.softmax(QK, dim=-1)
    W = W[..., :-1]
    attn = torch.einsum("hmqk,khmd->qhmd", W, V)
    return attn.reshape(q_len, -1)


class AttentionBlock(torch.nn.Module):
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
        # Only apply sliding window to every other layer
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

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
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

        # Get start position from cache
        start_pos = 0 if kv_cache is None else kv_cache[0].shape[0]

        # Apply rotary embeddings with correct positions
        q, k = self.rope(q, k, start_pos=start_pos)

        # Concatenate cached K, V with new K, V
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=0)
            v = torch.cat([kv_cache[1], v], dim=0)

        t = sdpa(q, k, v, self.sinks, self.sm_scale, self.sliding_window, start_pos=start_pos)
        t = self.out(t)
        t = x + t

        # FIX: Trim KV cache for sliding window layers to prevent unbounded memory growth.
        # Full-context layers (sliding_window == 0) keep the entire history.
        new_cache = None
        if use_cache:
            if self.sliding_window > 0:
                # Only the last `sliding_window` tokens are ever attended to,
                # so there is no benefit in retaining older entries.
                k_cache = k[-self.sliding_window :]
                v_cache = v[-self.sliding_window :]
            else:
                k_cache, v_cache = k, v
            new_cache = (k_cache, v_cache)

        return t, new_cache


def swiglu(x, alpha: float = 1.702, limit: float = 7.0):
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    # Clamp the input values
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    # Note we add an extra bias of 1 to the linear layer
    return out_glu * (x_linear + 1)


class MLPBlock(torch.nn.Module):
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
        self.mlp1_weight = torch.nn.Parameter(
            torch.empty(
                (
                    config.num_experts,
                    config.intermediate_size * 2 // self.world_size,
                    config.hidden_size,
                ),
                device=device,
                dtype=torch.bfloat16,
            )
        )
        self.mlp1_bias = torch.nn.Parameter(
            torch.empty(
                (config.num_experts, config.intermediate_size * 2 // self.world_size),
                device=device,
                dtype=torch.bfloat16,
            )
        )
        self.mlp2_weight = torch.nn.Parameter(
            torch.empty(
                (
                    config.num_experts,
                    config.hidden_size,
                    config.intermediate_size // self.world_size,
                ),
                device=device,
                dtype=torch.bfloat16,
            )
        )
        self.mlp2_bias = torch.nn.Parameter(
            torch.empty(
                (config.num_experts, config.hidden_size),
                device=device,
                dtype=torch.bfloat16,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.norm(x)
        g = self.gate(t)
        experts = torch.topk(g, k=self.experts_per_token, dim=-1, sorted=True)
        expert_weights = torch.nn.functional.softmax(experts.values, dim=1)
        expert_indices = experts.indices

        # MLP #1
        mlp1_weight = self.mlp1_weight[expert_indices, ...]
        mlp1_bias = self.mlp1_bias[expert_indices, ...]
        t = torch.einsum("beck,bk->bec", mlp1_weight, t) + mlp1_bias
        t = swiglu(t, limit=self.swiglu_limit)

        # MLP #2
        mlp2_weight = self.mlp2_weight[expert_indices, ...]
        mlp2_bias = self.mlp2_bias[expert_indices, ...]
        t = torch.einsum("beck,bek->bec", mlp2_weight, t)
        if self.world_size > 1:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t += mlp2_bias

        # Weighted sum of experts
        t = torch.einsum("bec,be->bc", t, expert_weights)

        return x + t


class TransformerBlock(torch.nn.Module):
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

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        x, new_cache = self.attn(x, kv_cache=kv_cache, use_cache=use_cache)
        x = self.mlp(x)
        return x, new_cache


class Transformer(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.embedding = torch.nn.Embedding(
            config.vocab_size, config.hidden_size, device=device, dtype=torch.bfloat16
        )
        self.block = torch.nn.ModuleList(
            [
                TransformerBlock(config, layer_idx, device)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, device=device)
        self.unembedding = torch.nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            device=device,
            dtype=torch.bfloat16,
        )

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]] | None]:
        x = self.embedding(x)

        new_kv_cache = [] if use_cache else None

        for i, block in enumerate(self.block):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            x, layer_new_cache = block(x, kv_cache=layer_cache, use_cache=use_cache)
            if use_cache:
                new_kv_cache.append(layer_new_cache)

        x = self.norm(x)
        x = self.unembedding(x)
        return x, new_kv_cache

    @classmethod
    def from_checkpoint(
        cls, path: str, device: str | torch.device = "cuda"
    ) -> "Transformer":
        if not isinstance(device, torch.device):
            device = torch.device(device)

        config_path = os.path.join(path, "config.json")
        with open(config_path, "r") as f:
            config = ModelConfig(**json.load(f))

        model = Transformer(config=config, device=device)
        model.eval()

        my_rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        checkpoint = Checkpoint(path, device)
        load_errors = []

        for name, param in model.named_parameters():
            loaded_tensor = checkpoint.get(name)

            # FIX: Use explicit name checks instead of "mlp1" in name to avoid
            # accidentally matching unrelated parameters in future refactors.

            # mlp1_weight: (num_experts, 2*intermediate_size, hidden_size)
            # mlp1_bias:   (num_experts, 2*intermediate_size)
            # Both are column-parallel: shard on dim 1 (the intermediate dim).
            if "mlp1_weight" in name or "mlp1_bias" in name:
                full_dim = loaded_tensor.shape[1]
                assert full_dim % world_size == 0, (
                    f"{name}: dim 1 size {full_dim} is not divisible by "
                    f"world_size {world_size}"
                )
                chunk = full_dim // world_size
                loaded_tensor = loaded_tensor[
                    :, my_rank * chunk : (my_rank + 1) * chunk, ...
                ]

            # mlp2_weight: (num_experts, hidden_size, intermediate_size)
            # Row-parallel: shard on dim 2 (the intermediate/input dim).
            # mlp2_bias:   (num_experts, hidden_size) — NOT sharded.
            # It is the same on every rank because it is added after all_reduce.
            elif "mlp2_weight" in name:
                full_dim = loaded_tensor.shape[2]
                assert full_dim % world_size == 0, (
                    f"{name}: dim 2 size {full_dim} is not divisible by "
                    f"world_size {world_size}"
                )
                chunk = full_dim // world_size
                loaded_tensor = loaded_tensor[
                    :, :, my_rank * chunk : (my_rank + 1) * chunk
                ]

            # Cast dtype if the checkpoint stores weights in a different precision
            # (e.g. MXFP4 / FP8 checkpoints upcasted to bfloat16 at load time).
            if loaded_tensor.dtype != param.data.dtype:
                loaded_tensor = loaded_tensor.to(param.data.dtype)

            # Collect shape mismatches rather than crashing on the first one so
            # the full picture is visible in a single error message.
            if loaded_tensor.shape != param.data.shape:
                load_errors.append(
                    f"  {name}: param={tuple(param.data.shape)}, "
                    f"loaded={tuple(loaded_tensor.shape)}"
                )
                continue

            param.data.copy_(loaded_tensor)

        if load_errors:
            raise RuntimeError(
                f"Shape mismatches loading checkpoint from '{path}':\n"
                + "\n".join(load_errors)
            )

        return model


class TokenGenerator:
    @torch.inference_mode()
    def __init__(self, checkpoint: str, device: torch.device):
        self.device = device
        self.model = Transformer.from_checkpoint(checkpoint, device=self.device)

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: list[int],
        stop_tokens: list[int],
        temperature: float = 1.0,
        max_tokens: int = 100,
        return_logprobs: bool = False,
    ):
        # First pass: process entire prompt, build KV cache
        input_tensor = torch.as_tensor(
            prompt_tokens, dtype=torch.int32, device=self.device
        )
        logits, kv_cache = self.model(input_tensor, use_cache=True)
        logits = logits[-1]  # Logits for last token

        num_generated_tokens = 0
        while num_generated_tokens < max_tokens:
            if temperature == 0.0:
                predicted_token = torch.argmax(logits, dim=-1).item()
            else:
                probs = torch.softmax(logits * (1.0 / temperature), dim=-1)
                predicted_token = torch.multinomial(probs, num_samples=1).item()

            num_generated_tokens += 1

            if return_logprobs:
                logprobs = torch.log_softmax(logits, dim=-1)
                selected_logprobs = logprobs[predicted_token].item()
                yield predicted_token, selected_logprobs
            else:
                yield predicted_token

            if predicted_token in stop_tokens:
                break

            # Process only the new token, reuse KV cache
            input_tensor = torch.as_tensor(
                [predicted_token], dtype=torch.int32, device=self.device
            )
            logits, kv_cache = self.model(
                input_tensor, kv_cache=kv_cache, use_cache=True
            )
            logits = logits[-1]

    @torch.inference_mode()
    def generate_with_cache(
        self,
        new_tokens: list[int],
        stop_tokens: list[int],
        kv_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        temperature: float = 1.0,
        max_tokens: int = 100,
    ) -> tuple[list[int], list[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Generate tokens with KV cache persistence.

        Args:
            new_tokens: New tokens to process (just the new part of the conversation).
            stop_tokens: Token IDs that signal end of generation.
            kv_cache: Existing KV cache from previous turns. None for new conversation.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            tuple: (generated_tokens, updated_kv_cache)
        """
        # Process new tokens, extending existing KV cache if provided
        input_tensor = torch.as_tensor(
            new_tokens, dtype=torch.int32, device=self.device
        )
        logits, kv_cache = self.model(input_tensor, kv_cache=kv_cache, use_cache=True)
        logits = logits[-1]  # Logits for last token

        generated_tokens = []
        num_generated_tokens = 0

        while num_generated_tokens < max_tokens:
            if temperature == 0.0:
                predicted_token = torch.argmax(logits, dim=-1).item()
            else:
                probs = torch.softmax(logits * (1.0 / temperature), dim=-1)
                predicted_token = torch.multinomial(probs, num_samples=1).item()

            num_generated_tokens += 1
            generated_tokens.append(predicted_token)

            if predicted_token in stop_tokens:
                break

            # Process only the new token, reuse KV cache
            input_tensor = torch.as_tensor(
                [predicted_token], dtype=torch.int32, device=self.device
            )
            logits, kv_cache = self.model(
                input_tensor, kv_cache=kv_cache, use_cache=True
            )
            logits = logits[-1]

        return generated_tokens, kv_cache

    def clear_cache(self):
        """Clear CUDA cache to free memory."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()