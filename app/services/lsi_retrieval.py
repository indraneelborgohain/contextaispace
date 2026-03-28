"""
Semantic turn retrieval for KV cache context assembly.

Algorithm:
  1. Encode the current query with MiniLM (384-dim, L2-normalised).
  2. Compare against each stored turn's pre-computed embedding via
     cosine similarity (dot product on normalised vectors).
  3. Always pin the most recent turn (index -1).
  4. Greedily fill the remaining token budget with the top-K most
     similar turns (highest cosine sim first).
  5. Return selected indices in chronological order so Re-RoPE stays
     contiguous.

No SVD, no BoW matrices — just embedding cosine similarity.
"""

import numpy as np
from typing import List, Tuple

from app.services.kv_cache_store import TurnDelta
from app.config import TOP_K_TURNS


def select_turns(
    turns: List[TurnDelta],
    query_text: str,
    token_budget: int,
    top_k: int = TOP_K_TURNS,
) -> List[int]:
    """Select the most relevant past turns to fill *token_budget* tokens.

    Args:
        turns:        All stored TurnDelta objects for this conversation,
                      in chronological order (index 0 = oldest).
        query_text:   The current user query as plain text.
        token_budget: Maximum total tokens the selected turns may occupy.
        top_k:        Maximum number of similar turns to consider beyond
                      the always-pinned most-recent turn.

    Returns:
        Sorted list of turn indices (ascending / chronological) whose
        combined num_tokens fits within *token_budget*.
    """
    if not turns:
        return []

    from app.services.embedder import get_embedder
    embedder = get_embedder()

    # ------------------------------------------------------------------
    # 1. Embed the current query
    # ------------------------------------------------------------------
    query_emb = embedder.encode(query_text)   # [384], already L2-normalised

    # ------------------------------------------------------------------
    # 2. Score every past turn by cosine similarity to the query
    # ------------------------------------------------------------------
    scores: List[Tuple[int, float]] = []
    for idx, turn in enumerate(turns):
        sim = embedder.cosine_similarity(query_emb, turn.embedding)
        scores.append((idx, sim))

    # Sort descending by similarity
    scores.sort(key=lambda x: x[1], reverse=True)

    # ------------------------------------------------------------------
    # 3. Always pin the most recent turn (last index)
    # ------------------------------------------------------------------
    pinned_idx   = len(turns) - 1
    pinned_turn  = turns[pinned_idx]
    selected     = set()
    remaining    = token_budget

    if pinned_turn.num_tokens <= remaining:
        selected.add(pinned_idx)
        remaining -= pinned_turn.num_tokens

    # ------------------------------------------------------------------
    # 4. Greedily fill with top-K similar turns (skip pinned)
    # ------------------------------------------------------------------
    added = 0
    for idx, _sim in scores:
        if added >= top_k:
            break
        if idx in selected:
            continue
        if turns[idx].num_tokens <= remaining:
            selected.add(idx)
            remaining -= turns[idx].num_tokens
            added += 1

    # ------------------------------------------------------------------
    # 5. Return in chronological order
    # ------------------------------------------------------------------
    return sorted(selected)
