"""
Filesystem-based chat history.

Stores one JSON file per conversation under data/chat_history/.
Each file is a list of message dicts:
    { role, content, intent, continues, timestamp }

This replaces the old single-file chat_history.json approach and the
old ChatHistoryService class. Swap this module for a Supabase adapter
later without touching anything else.
"""

import json
import os
import time
import threading
from typing import List, Dict, Optional

from app.config import HISTORY_DIR


class ChatHistoryFS:
    """Persist conversation messages to individual JSON files."""

    def __init__(self, history_dir: str = HISTORY_DIR):
        self._dir  = history_dir
        self._lock = threading.Lock()
        os.makedirs(self._dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_turn(
        self,
        conv_id: str,
        role: str,
        content: str,
        intent: str = "none",
        continues: bool = False,
    ) -> None:
        """Append a single message to *conv_id*'s history file."""
        message = {
            "role":      role,
            "content":   content,
            "intent":    intent,
            "continues": continues,
            "timestamp": time.time(),
        }
        with self._lock:
            messages = self._load(conv_id)
            messages.append(message)
            self._save(conv_id, messages)

    def get_messages(self, conv_id: str) -> List[Dict]:
        """Return all messages for *conv_id* (empty list if none)."""
        with self._lock:
            return self._load(conv_id)

    def list_conversations(self) -> List[str]:
        """Return all conversation IDs that have a history file."""
        with self._lock:
            return [
                f[:-5] for f in os.listdir(self._dir) if f.endswith(".json")
            ]

    def clear(self, conv_id: str) -> bool:
        """Delete history for *conv_id*. Returns True if it existed."""
        path = self._path(conv_id)
        with self._lock:
            if os.path.exists(path):
                os.remove(path)
                return True
            return False

    def clear_all(self) -> int:
        """Delete all history files. Returns count deleted."""
        with self._lock:
            count = 0
            for fname in os.listdir(self._dir):
                if fname.endswith(".json"):
                    os.remove(os.path.join(self._dir, fname))
                    count += 1
            return count

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _path(self, conv_id: str) -> str:
        # Sanitise conv_id to be a safe filename
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in conv_id)
        return os.path.join(self._dir, f"{safe}.json")

    def _load(self, conv_id: str) -> List[Dict]:
        path = self._path(conv_id)
        if not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []

    def _save(self, conv_id: str, messages: List[Dict]) -> None:
        path = self._path(conv_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_history: Optional[ChatHistoryFS] = None


def get_chat_history() -> ChatHistoryFS:
    """Return the process-global ChatHistoryFS, creating it on first call."""
    global _history
    if _history is None:
        _history = ChatHistoryFS()
    return _history
