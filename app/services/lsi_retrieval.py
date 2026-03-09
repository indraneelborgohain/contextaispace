"""
Latent Semantic Indexing (LSI) retrieval for selecting the most relevant
past conversation turns to include in the KV cache budget.

Given a set of per-turn bag-of-words vectors and a query BoW vector,
this module:
  1. Builds a term-document matrix from all past turns.
  2. Computes a truncated SVD (the LSI decomposition).
  3. Projects the current query into the low-rank topic space.
  4. Ranks past turns by cosine similarity to the query.
  5. Greedily selects turns (most-relevant first) until the token
     budget is filled, always including turn 0 (system prompt).
"""

import numpy as np
from typing import List, Tuple

from .kv_cache_store import TurnDelta


def build_bow_vector(token_ids: List[int], vocab_size: int) -> np.ndarray:
    """Create a bag-of-words (term-frequency) vector from token ids."""
    vec = np.zeros(vocab_size, dtype=np.float32)
    for tid in token_ids:
        if 0 <= tid < vocab_size:
            vec[tid] += 1.0
    return vec


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def select_turns_lsi(
    turns: List[TurnDelta],
    query_token_ids: List[int],
    vocab_size: int,
    token_budget: int,
    lsi_rank: int = 256,
) -> List[int]:
    """Select the most relevant past turns via LSI to fill *token_budget*.

    Args:
        turns: All stored per-turn deltas for a conversation.
        query_token_ids: Tokenized current user query.
        vocab_size: Model vocabulary size (for BoW dimension).
        token_budget: Maximum total tokens the selected turns may occupy.
        lsi_rank: Number of SVD components for the topic space.  Higher
            values preserve more nuance but cost more compute.

    Returns:
        Sorted list of turn indices to include (ascending order, turn 0
        always first if it fits).
    """
    if not turns:
        return []

    num_turns = len(turns)

    # ------------------------------------------------------------------
    # 1. Build term-document matrix  (vocab_size × num_turns)
    # ------------------------------------------------------------------
    td_matrix = np.column_stack([t.bow_vector for t in turns])  # (V, T)

    # Apply TF-IDF weighting: log(1 + tf) * idf
    tf = np.log1p(td_matrix)
    doc_freq = np.sum(td_matrix > 0, axis=1, keepdims=True).astype(np.float32)
    doc_freq = np.maximum(doc_freq, 1.0)  # avoid div-by-zero
    idf = np.log(num_turns / doc_freq + 1.0)
    td_matrix = tf * idf

    # ------------------------------------------------------------------
    # 2. Truncated SVD  →  U (V×k), S (k,), Vt (k×T)
    # ------------------------------------------------------------------
    rank = min(lsi_rank, num_turns, td_matrix.shape[0])
    # numpy full SVD then truncate (lightweight for small T)
    U, S, Vt = np.linalg.svd(td_matrix, full_matrices=False)
    U_k = U[:, :rank]       # (V, k)
    S_k = S[:rank]           # (k,)
    Vt_k = Vt[:rank, :]     # (k, T)

    # Turn representations in topic space:  columns of Vt_k transposed
    turn_topics = Vt_k.T    # (T, k)

    # ------------------------------------------------------------------
    # 3. Project query into topic space
    # ------------------------------------------------------------------
    query_bow = build_bow_vector(query_token_ids, vocab_size).astype(np.float32)
    query_tf = np.log1p(query_bow)
    query_tfidf = query_tf * idf.squeeze()
    # project:  q_topic = query_tfidf @ U_k @ diag(1/S_k)
    S_inv = np.where(S_k > 1e-10, 1.0 / S_k, 0.0)
    query_topic = (query_tfidf @ U_k) * S_inv   # (k,)

    # ------------------------------------------------------------------
    # 4. Rank turns by cosine similarity
    # ------------------------------------------------------------------
    similarities: List[Tuple[int, float]] = []
    for idx in range(num_turns):
        sim = _cosine_similarity(query_topic, turn_topics[idx])
        similarities.append((idx, sim))

    # Sort descending by similarity (turn 0 gets special treatment below)
    similarities.sort(key=lambda x: x[1], reverse=True)

    # ------------------------------------------------------------------
    # 5. Greedy selection: always include turn 0, then fill by rank
    # ------------------------------------------------------------------
    selected: List[int] = []
    remaining_budget = token_budget

    # Always include turn 0 (system prompt) if it fits
    if turns[0].num_tokens <= remaining_budget:
        selected.append(0)
        remaining_budget -= turns[0].num_tokens

    for idx, _sim in similarities:
        if idx == 0:
            continue  # already handled
        if remaining_budget <= 0:
            break
        if turns[idx].num_tokens <= remaining_budget:
            selected.append(idx)
            remaining_budget -= turns[idx].num_tokens

    # Return in chronological order so positional re-roping is contiguous
    selected.sort()
    return selected
