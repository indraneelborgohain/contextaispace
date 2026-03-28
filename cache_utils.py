import torch
import numpy as np
from typing import List, Tuple, Optional

from architecture.gptoss20B import ModelConfig, reposition_rope


# ---------------------------------------------------------------------------
# Context-window budget layout  (configurable)
# ---------------------------------------------------------------------------
context_len: int = ModelConfig.initial_context_length
prefix_budget: int = 3096   # First N positions kept verbatim (system prompt)
lsi_budget: int = 500       # Filled by LSI-retrieved past turn deltas

# Legacy SVD budget kept as fallback (unused in LSI path)
svd_budget: int = 100

# Even layers use sliding-window (max 128 tokens); odd layers keep full context.
# Always use an odd layer for cache-length measurements.
_FULL_CTX_LAYER: int = 1


def _trim_cache(
    kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
    n: int = 1
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Remove last n token positions from every layer of the KV cache."""
    return [(k[:-n], v[:-n]) for k, v in kv_cache]

def _svd_compress_cache(
    kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
    max_len: int = context_len,
    budget: int = svd_budget
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Compress KV cache when it exceeds max_len using SVD on the overflow.

    Legacy fallback — the primary path now uses LSI retrieval via
    ``assemble_cache_with_lsi``.

    Strategy:
      - Keep the first (max_len - budget) positions as-is.
      - Compress the remaining overflow positions into `budget` virtual tokens
        via truncated SVD.
      - Final cache size = max_len.
    """
    if kv_cache is None:
        return None
    current_len = kv_cache[_FULL_CTX_LAYER][0].shape[0]
    if current_len <= max_len:
        return kv_cache

    keep_len = max_len - budget

    compressed = []
    for k, v in kv_cache:
        layer_len = k.shape[0]
        if layer_len <= keep_len:
            # Sliding-window layer — shorter than keep_len, nothing to compress
            compressed.append((k, v))
            continue

        k_keep = k[:keep_len]
        v_keep = v[:keep_len]

        overflow_k = k[keep_len:]
        overflow_v = v[keep_len:]
        if overflow_k.shape[0] == 0:
            compressed.append((k_keep, v_keep))
            continue

        k_compressed = _svd_compress_tensor(overflow_k, budget)
        v_compressed = _svd_compress_tensor(overflow_v, budget)

        compressed.append((
            torch.cat([k_keep, k_compressed], dim=0),
            torch.cat([v_keep, v_compressed], dim=0),
        ))

    return compressed


def _svd_compress_tensor(tensor: torch.Tensor, budget: int) -> torch.Tensor:
    """Compress sequence positions via truncated SVD (legacy helper)."""
    orig_shape = tensor.shape
    seq_len = orig_shape[0]
    rest_shape = orig_shape[1:]
    hidden = 1
    for d in rest_shape:
        hidden *= d

    mat = tensor.reshape(seq_len, hidden).float()
    rank = min(budget, seq_len, hidden)

    U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
    compressed = torch.diag(S[:rank]) @ Vh[:rank]

    if rank < budget:
        pad = torch.zeros(
            budget - rank, hidden, dtype=compressed.dtype, device=compressed.device
        )
        compressed = torch.cat([compressed, pad], dim=0)

    compressed = compressed.to(tensor.dtype).reshape(budget, *rest_shape)
    return compressed


# ---------------------------------------------------------------------------
# LSI-based cache assembly
# ---------------------------------------------------------------------------

def _extract_turn_delta(
    kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
    start: int,
    end: int,
) -> List[Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    """Slice a range of positions from a cumulative KV cache.

    For sliding-window layers whose cache is shorter than ``end``,
    stores None (these layers are ephemeral and don't need LSI retrieval).
    """
    deltas = []
    for i, (k, v) in enumerate(kv_cache):
        if k.shape[0] < end or start >= end:
            deltas.append(None)
        else:
            deltas.append((k[start:end].clone(), v[start:end].clone()))
    return deltas


def _build_bow(token_ids: List[int], vocab_size: int) -> np.ndarray:
    """Bag-of-words vector from token ids."""
    vec = np.zeros(vocab_size, dtype=np.float32)
    for tid in token_ids:
        if 0 <= tid < vocab_size:
            vec[tid] += 1.0
    return vec


def assemble_cache_with_lsi(
    recent_kv: List[Tuple[torch.Tensor, torch.Tensor]],
    past_turns: list,          # List[TurnDelta]
    query_token_ids: List[int],
    generator,                 # TokenGenerator — for rope modules
    vocab_size: int = ModelConfig.vocab_size,
    prefix_len: int = prefix_budget,
    lsi_len: int = lsi_budget,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Assemble a context-window-sized KV cache.

    Layout:  [first prefix_len positions (kept verbatim)] + [LSI-retrieved past turns (≤ lsi_len)]

    The first prefix_len positions preserve the system prompt and early
    conversation.  LSI-selected past turn deltas are re-RoPE'd and
    appended after the prefix.

    Args:
        recent_kv: Current cumulative KV cache (may exceed context_len).
        past_turns: Per-turn deltas from KVCacheStore.
        query_token_ids: Current user query tokens (for LSI ranking).
        generator: TokenGenerator whose model has RoPE modules.
        vocab_size: Vocabulary size for BoW vectors.
        prefix_len: Tokens kept verbatim from the start of the cache.
        lsi_len: Token budget for LSI-retrieved turns.

    Returns:
        Assembled KV cache of size ≤ context_len.
    """
    from app.services.lsi_retrieval import select_turns_lsi

    current_len = recent_kv[_FULL_CTX_LAYER][0].shape[0]

    # If still within context window, nothing to do
    if current_len <= context_len:
        return recent_kv

    # --- A. Keep the first prefix_len positions (system prompt + early context) ---
    actual_prefix = min(prefix_len, current_len)
    prefix_slice = []
    for k, v in recent_kv:
        layer_len = k.shape[0]
        layer_prefix = min(actual_prefix, layer_len)
        prefix_slice.append((k[:layer_prefix], v[:layer_prefix]))

    # --- B. Filter to only turns beyond the prefix ----------------------
    # Turns whose tokens are entirely within the first prefix_len positions
    # are already preserved verbatim — no need to select them via LSI.
    overflow_turns = [
        t for t in past_turns
        if t.start_pos + t.num_tokens > actual_prefix
    ]

    # --- C. If no overflow turns stored yet, fall back to SVD ---------
    if not overflow_turns:
        return _svd_compress_cache(recent_kv, max_len=context_len, budget=lsi_len)

    # --- D. Select from overflow turns via LSI ------------------------
    selected_indices = select_turns_lsi(
        turns=overflow_turns,
        query_token_ids=query_token_ids,
        vocab_size=vocab_size,
        token_budget=lsi_len,
    )

    if not selected_indices:
        # Nothing selected — just truncate to context_len
        return [(k[:context_len], v[:context_len]) for k, v in recent_kv]

    # --- D. Concatenate selected turn deltas & re-RoPE ----------------
    num_layers = len(recent_kv)
    assembled_k_parts: List[List[torch.Tensor]] = [[] for _ in range(num_layers)]
    assembled_v_parts: List[List[torch.Tensor]] = [[] for _ in range(num_layers)]

    # LSI turns are placed right after the prefix
    new_pos_offset = actual_prefix

    # Access rope modules from model layers
    rope_modules = []
    for block in generator.model.block:
        rope_modules.append(block.attn.rope)

    for turn_idx in selected_indices:
        turn = overflow_turns[turn_idx]
        turn_len = turn.num_tokens

        if turn_len == 0:
            continue

        old_positions = torch.arange(
            turn.start_pos, turn.start_pos + turn_len,
            device=generator.device,
        )
        new_positions = torch.arange(
            new_pos_offset, new_pos_offset + turn_len,
            device=generator.device,
        )

        for layer_idx in range(num_layers):
            delta = turn.kv_delta[layer_idx]
            if delta is None:
                # Sliding-window layer — no delta stored; skip
                continue
            k_delta, v_delta = delta
            k_delta = k_delta.to(generator.device)
            v_delta = v_delta.to(generator.device)

            # Re-RoPE the K cache to new contiguous positions
            k_repositioned = reposition_rope(
                k_delta, old_positions, new_positions, rope_modules[layer_idx]
            )
            assembled_k_parts[layer_idx].append(k_repositioned)
            assembled_v_parts[layer_idx].append(v_delta)

        new_pos_offset += turn_len

    # --- E. Assemble: [prefix | LSI turns] ----------------------------
    final_cache = []
    for layer_idx in range(num_layers):
        k_prefix, v_prefix = prefix_slice[layer_idx]

        if assembled_k_parts[layer_idx]:
            # Full-context layer: append LSI turns after prefix
            all_k = [k_prefix] + assembled_k_parts[layer_idx]
            all_v = [v_prefix] + assembled_v_parts[layer_idx]

            final_cache.append((
                torch.cat(all_k, dim=0),
                torch.cat(all_v, dim=0),
            ))
        else:
            # Sliding-window layer: just keep prefix
            final_cache.append((k_prefix, v_prefix))

    return final_cache
