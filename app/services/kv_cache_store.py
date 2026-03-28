"""
KV Cache Store — in-memory, per-conversation storage of KV deltas.

Design:
  - Each conversation holds a list of TurnDelta objects.
  - Each TurnDelta stores ONLY the KV slice for that turn (delta),
    plus a MiniLM sentence embedding for semantic similarity retrieval.
  - Two-level memory budget:
      1. Per-conversation cap  (MAX_TURNS_PER_USER, MAX_TOKENS_PER_USER)
      2. Global pool cap       (MAX_TOTAL_TURNS)
  - Eviction: LRU at conversation level; oldest-turn at turn level.
  - On eviction the KV tensors are freed; token_ids stay so KV can be
    recomputed from text on a cold path if needed.
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from app.config import (
    MAX_TURNS_PER_USER,
    MAX_TOKENS_PER_USER,
    MAX_TOTAL_TURNS,
    TTL_SECONDS,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TurnDelta:
    """All data stored for a single conversation turn."""

    # KV cache slice for this turn only — list[Optional[(K, V)]] per layer.
    # None for sliding-window layers (they are ephemeral by design).
    kv_delta: List[Optional[Tuple[torch.Tensor, torch.Tensor]]]

    # Raw token ids for the full turn (user msg + assistant response).
    # Kept so we can recompute KV on a cold start.
    token_ids: List[int]

    # MiniLM sentence embedding of the turn text — used for similarity search.
    embedding: np.ndarray          # shape [384]

    # Positional metadata needed for Re-RoPE when assembling context.
    start_pos: int                 # absolute position in the conversation stream
    num_tokens: int                # number of tokens in this turn

    # Labels produced by the model for this turn.
    intent: str = "none"
    continues: bool = False

    # Whether the KV tensors are currently resident in memory.
    kv_loaded: bool = True


@dataclass
class ConversationCache:
    """Per-conversation container."""
    turns: List[TurnDelta] = field(default_factory=list)
    total_tokens: int = 0          # sum of num_tokens across all turns
    last_access: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class KVCacheStore:
    """
    Thread-safe in-memory store for per-turn KV cache deltas.

    Public API:
        add_turn(conv_id, turn_delta)  — append a new turn
        get_turns(conv_id)             — return list of TurnDelta
        clear(conv_id)                 — free a conversation
        clear_all()                    — free everything
        stats()                        — dict of memory usage
    """

    def __init__(
        self,
        max_turns_per_conv: int = MAX_TURNS_PER_USER,
        max_tokens_per_conv: int = MAX_TOKENS_PER_USER,
        max_total_turns: int = MAX_TOTAL_TURNS,
        ttl_seconds: int = TTL_SECONDS,
    ):
        self._max_turns_per_conv  = max_turns_per_conv
        self._max_tokens_per_conv = max_tokens_per_conv
        self._max_total_turns     = max_total_turns
        self._ttl_seconds         = ttl_seconds
        self._store: Dict[str, ConversationCache] = {}
        self._lock  = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_turn(self, conv_id: str, turn: TurnDelta) -> None:
        """Append *turn* to *conv_id*, enforcing memory budgets."""
        with self._lock:
            self._expire_stale()

            if conv_id not in self._store:
                self._maybe_evict_lru_conversation()
                self._store[conv_id] = ConversationCache()

            cache = self._store[conv_id]
            cache.turns.append(turn)
            cache.total_tokens += turn.num_tokens
            cache.last_access = time.time()

            self._enforce_per_conv_limits(cache)

    def get_turns(self, conv_id: str) -> List[TurnDelta]:
        """Return the list of TurnDeltas for *conv_id* (empty if missing/expired)."""
        with self._lock:
            cache = self._store.get(conv_id)
            if cache is None:
                return []
            if time.time() - cache.last_access > self._ttl_seconds:
                self._free_conversation(conv_id)
                return []
            cache.last_access = time.time()
            return list(cache.turns)

    def clear(self, conv_id: str) -> bool:
        """Free a single conversation. Returns True if it existed."""
        with self._lock:
            return self._free_conversation(conv_id)

    def clear_all(self) -> int:
        """Free all conversations. Returns count freed."""
        with self._lock:
            ids = list(self._store.keys())
            for cid in ids:
                self._free_conversation(cid)
            return len(ids)

    def stats(self) -> dict:
        with self._lock:
            total_turns  = sum(len(c.turns) for c in self._store.values())
            total_tokens = sum(c.total_tokens for c in self._store.values())
            return {
                "num_conversations": len(self._store),
                "total_turns_in_memory": total_turns,
                "total_tokens_in_memory": total_tokens,
                "max_turns_per_conv": self._max_turns_per_conv,
                "max_tokens_per_conv": self._max_tokens_per_conv,
                "max_total_turns": self._max_total_turns,
            }

    def list_conversations(self) -> List[Dict]:
        """List all cached conversations with metadata."""
        with self._lock:
            result = []
            for conv_id, cache in self._store.items():
                result.append({
                    "conversation_id": conv_id,
                    "num_turns": len(cache.turns),
                    "total_tokens": cache.total_tokens,
                    "last_access": cache.last_access,
                })
            return sorted(result, key=lambda x: x["last_access"], reverse=True)

    # ------------------------------------------------------------------
    # Internal helpers  (caller must hold self._lock)
    # ------------------------------------------------------------------

    def _enforce_per_conv_limits(self, cache: ConversationCache) -> None:
        """Evict oldest turns from *cache* until within per-conv budget."""
        while (
            len(cache.turns) > self._max_turns_per_conv
            or cache.total_tokens > self._max_tokens_per_conv
        ):
            if not cache.turns:
                break
            evicted = cache.turns.pop(0)
            cache.total_tokens -= evicted.num_tokens
            self._free_turn_kv(evicted)

    def _maybe_evict_lru_conversation(self) -> None:
        """If global pool is full, evict the least-recently-used conversation."""
        total = sum(len(c.turns) for c in self._store.values())
        if total < self._max_total_turns or not self._store:
            return
        lru_id = min(self._store, key=lambda k: self._store[k].last_access)
        self._free_conversation(lru_id)

    def _expire_stale(self) -> None:
        now = time.time()
        stale = [
            cid for cid, c in self._store.items()
            if now - c.last_access > self._ttl_seconds
        ]
        for cid in stale:
            self._free_conversation(cid)

    def _free_conversation(self, conv_id: str) -> bool:
        cache = self._store.pop(conv_id, None)
        if cache is None:
            return False
        for turn in cache.turns:
            self._free_turn_kv(turn)
        return True

    @staticmethod
    def _free_turn_kv(turn: TurnDelta) -> None:
        """Delete KV tensors and mark the turn as unloaded."""
        if turn.kv_delta:
            for entry in turn.kv_delta:
                if entry is not None:
                    del entry[0], entry[1]
        turn.kv_delta  = []
        turn.kv_loaded = False


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_store: Optional[KVCacheStore] = None


def get_kv_cache_store() -> KVCacheStore:
    """Return the process-global KVCacheStore, creating it on first call."""
    global _store
    if _store is None:
        _store = KVCacheStore()
    return _store
