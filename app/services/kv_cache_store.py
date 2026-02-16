"""
KV Cache Store

In-memory storage for KV caches per conversation to enable efficient
continuation of multi-turn conversations without reprocessing history.

TODO: For scaling to many concurrent users, consider:
  - Reduce max_conversations or TTL
  - Move to distributed cache (Redis + tensor serialization)
  - Implement cache sharding across multiple GPU nodes
"""

import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import threading
import time


@dataclass
class ConversationCache:
    """Stores KV cache and token history for a conversation."""
    kv_cache: List[Tuple[torch.Tensor, torch.Tensor]]  # Per-layer (K, V) tuples
    tokens_so_far: List[int]  # All tokens processed so far
    last_access: float  # Timestamp for cache eviction
    

class KVCacheStore:
    """
    In-memory store for KV caches per conversation.
    
    Manages KV cache lifecycle:
    - Store cache after each response
    - Retrieve cache for continuing conversations
    - Clear cache when conversation is reset
    - Auto-evict old caches to manage memory
    """
    
    def __init__(self, max_conversations: int = 100, ttl_seconds: int = 3600):
        """
        Initialize the KV cache store.
        
        Args:
            max_conversations: Maximum number of conversations to cache.
            ttl_seconds: Time-to-live for cache entries (default: 1 hour).
        """
        self.max_conversations = max_conversations
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, ConversationCache] = {}
        self._lock = threading.Lock()
    
    def get(self, conversation_id: str) -> Optional[ConversationCache]:
        """
        Get the cached data for a conversation.
        
        Args:
            conversation_id: Unique conversation identifier.
        
        Returns:
            ConversationCache or None if not found/expired.
        """
        with self._lock:
            if conversation_id not in self._cache:
                return None
            
            cache = self._cache[conversation_id]
            
            # Check TTL
            if time.time() - cache.last_access > self.ttl_seconds:
                self._remove_cache(conversation_id)
                return None
            
            # Update access time
            cache.last_access = time.time()
            return cache
    
    def set(
        self,
        conversation_id: str,
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        tokens_so_far: List[int]
    ) -> None:
        """
        Store or update cache for a conversation.
        
        Args:
            conversation_id: Unique conversation identifier.
            kv_cache: The KV cache from the model.
            tokens_so_far: All tokens processed in this conversation.
        """
        with self._lock:
            # Evict old entries if at capacity
            if conversation_id not in self._cache and len(self._cache) >= self.max_conversations:
                self._evict_oldest()
            
            self._cache[conversation_id] = ConversationCache(
                kv_cache=kv_cache,
                tokens_so_far=tokens_so_far,
                last_access=time.time()
            )
    
    def clear(self, conversation_id: str) -> bool:
        """
        Clear cache for a specific conversation.
        
        Args:
            conversation_id: Unique conversation identifier.
        
        Returns:
            True if cache was cleared, False if not found.
        """
        with self._lock:
            return self._remove_cache(conversation_id)
    
    def clear_all(self) -> int:
        """
        Clear all cached conversations.
        
        Returns:
            Number of caches cleared.
        """
        with self._lock:
            count = len(self._cache)
            for conv_id in list(self._cache.keys()):
                self._remove_cache(conv_id)
            return count
    
    def _remove_cache(self, conversation_id: str) -> bool:
        """Remove cache and free GPU memory."""
        if conversation_id in self._cache:
            cache = self._cache[conversation_id]
            # Clear tensor references to allow garbage collection
            if cache.kv_cache:
                for k, v in cache.kv_cache:
                    del k, v
            del self._cache[conversation_id]
            return True
        return False
    
    def _evict_oldest(self) -> None:
        """Evict the least recently accessed cache."""
        if not self._cache:
            return
        
        oldest_id = min(self._cache.keys(), key=lambda k: self._cache[k].last_access)
        self._remove_cache(oldest_id)
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        with self._lock:
            total_tokens = sum(len(c.tokens_so_far) for c in self._cache.values())
            return {
                "num_conversations": len(self._cache),
                "max_conversations": self.max_conversations,
                "total_tokens_cached": total_tokens,
                "ttl_seconds": self.ttl_seconds
            }
    
    def list_conversations(self) -> List[Dict]:
        """List all cached conversations with metadata."""
        with self._lock:
            conversations = []
            for conv_id, cache in self._cache.items():
                conversations.append({
                    "conversation_id": conv_id,
                    "num_tokens": len(cache.tokens_so_far),
                    "last_access": cache.last_access,
                    "num_layers_cached": len(cache.kv_cache) if cache.kv_cache else 0
                })
            return sorted(conversations, key=lambda x: x["last_access"], reverse=True)


# Singleton instance
_kv_cache_store: Optional[KVCacheStore] = None


def get_kv_cache_store(max_conversations: int = 100, ttl_seconds: int = 3600) -> KVCacheStore:
    """
    Get or create the singleton KVCacheStore instance.
    
    Args:
        max_conversations: Maximum conversations to cache (only used on first call).
        ttl_seconds: TTL for cache entries (only used on first call).
    
    Returns:
        KVCacheStore instance.
    """
    global _kv_cache_store
    if _kv_cache_store is None:
        _kv_cache_store = KVCacheStore(max_conversations, ttl_seconds)
    return _kv_cache_store
