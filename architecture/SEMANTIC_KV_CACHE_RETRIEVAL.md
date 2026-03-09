# Semantic KV Cache Retrieval (Turn-Level)

**Status:** Design complete, not implemented  
**Date:** February 22, 2026

## Overview

A memory-efficient approach to handle conversations exceeding the context window (4096 tokens) by using semantic similarity to retrieve relevant past turns instead of naive truncation.

## Core Idea

Instead of storing one cumulative KV cache that grows unbounded, store **per-turn deltas** with a **context vector** (last hidden state). When cache exceeds context window, use LSI/cosine similarity to select the most relevant past turns.

## Data Structure

```python
@dataclass
class TurnCache:
    kv_cache: List[Tuple[torch.Tensor, torch.Tensor]]  # Only THIS turn's K,V (delta)
    context_vector: torch.Tensor  # Last hidden state of this turn
    start_pos: int  # Original starting position
    num_tokens: int  # Number of tokens in this turn
```

## Flow

### Storage (after each turn):
```
Turn 1: 150 tokens → store kv_cache_1 (150), context_vector_1, pos 0-149
Turn 2: 200 tokens → store kv_cache_2 (200), context_vector_2, pos 150-349
Turn 3: 180 tokens → store kv_cache_3 (180), context_vector_3, pos 350-529
```

### Retrieval (when total > 4096):
```
1. Compute context_vector for current input
2. LSI/cosine similarity against all stored context_vectors
3. Select top-K turns that fit in budget (4096 - current_turn_tokens)
4. Always include Turn 1 (system prompt)
5. Re-rope selected K caches to contiguous positions
6. Concatenate and run inference
```

## Re-RoPE Algorithm

RoPE is reversible. To change a K cache from position `old_pos` to `new_pos`:

```python
def reposition_rope(k_cache, old_positions, new_positions, rope_module):
    # Get rotations for old positions
    cos_old, sin_old = rope_module._compute_cos_sin(old_positions)
    
    # Un-rotate (negate sin)
    k_unrotated = apply_rotary_emb(k_cache, cos_old, -sin_old)
    
    # Get rotations for new positions
    cos_new, sin_new = rope_module._compute_cos_sin(new_positions)
    
    # Apply new rotation
    k_renumbered = apply_rotary_emb(k_unrotated, cos_new, sin_new)
    
    return k_renumbered
```

**Note:** V cache has no RoPE, so it needs no modification.

## Example

```
Session has 20 turns, total 5000 tokens (exceeds 4096)

User asks: "What's my name?"

LSI query with current context_vector finds:
  - Turn 1 (system prompt): "You are helpful..." - 100 tokens
  - Turn 3: "My name is Alice" - 150 tokens  
  - Turn 15: "Remember I'm Alice" - 120 tokens
  - Current turn: 80 tokens

Total: 100 + 150 + 120 + 80 = 450 tokens ✓

Re-rope:
  Turn 1:  pos 0-99   (unchanged)
  Turn 3:  pos 100-249 (was 300-449)
  Turn 15: pos 250-369 (was 2100-2219)
  Current: pos 370-449

Run inference with 450-token combined cache.
```

## Key Insight

The context_vector of each turn captures "what this turn knew about." If Turn 3 mentioned "Alice", its context_vector encodes that knowledge. When the user asks about "name", LSI naturally retrieves Turn 3 because the vectors are semantically similar.

## Trade-offs

| Aspect | Pro | Con |
|--------|-----|-----|
| Memory | Delta storage, not cumulative | Extra storage for context vectors |
| Quality | Semantic > recency-based | May miss turns without direct textual match |
| Compute | Re-rope is cheap | LSI adds small latency |
| Complexity | Manageable | More state per conversation |

## Files to Modify

1. `app/services/kv_cache_store.py` - Change storage structure to per-turn
2. `architecture/gptoss20B.py` - Add `reposition_rope()` utility
3. `inference.py` - Modify `generateResultsWithCache()` to use retrieval
4. New file: `services/turn_retrieval.py` - LSI logic

## Why Delta Storage, Not Cumulative?

**Cumulative approach:**
```
Turn 1:  150 tokens → kv_cache_1 = 150 tokens
Turn 2:  200 tokens → kv_cache_2 = 350 tokens (includes turn 1)
Turn 15: 300 tokens → kv_cache_15 = 3000 tokens (includes turns 1-14)
```

**Problems with cumulative:**

1. **Massive overlap/duplication** - Turn 15 already contains Turn 1's KV. Picking both = tokens 0-149 appear twice.

2. **Memory explosion** - Storage grows O(N²) instead of O(N).

3. **No selective retrieval** - If you want ONLY Turn 1 + Turn 18:
   - Cumulative: Turn 18 cache = 4200 tokens (contains ALL turns 1-18, forced)
   - Delta: Turn 1 (150) + Turn 18 (300) = 450 tokens ✓

4. **Later turns are huge** - Turn 20's cumulative cache may exceed context window by itself.

**Delta is better because:**
- Pick any subset of turns independently
- No redundant KV entries
- Linear memory growth
- Maximum flexibility for semantic retrieval

**Key insight:** Retrieved KVs are just queried by current turn's Q. They don't cross-attend among themselves, so there's no benefit to keeping them "together" as cumulative.

## References

- Memorizing Transformers (Google)
- Landmark Attention
- Unlimiformer
- YaRN RoPE scaling (already in model)
