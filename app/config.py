"""
Central configuration for the GPT-OSS chat application.

All tuneable knobs live here. No magic numbers elsewhere.
"""

import os

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
PROJECT_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CHECKPOINT = os.path.join(PROJECT_ROOT, "model", "gpt-oss-20b", "original")

# ---------------------------------------------------------------------------
# Context window budgets
# ---------------------------------------------------------------------------
CONTEXT_LIMIT   : int = 4096   # Hard model limit (tokens)
LAST_N_BUDGET   : int = 1000   # Tokens reserved for the most recent turn(s)
TOP_K_TURNS     : int = 5      # Max similar past turns retrieved for context fill

# ---------------------------------------------------------------------------
# Memory management (in-memory KV cache)
# ---------------------------------------------------------------------------
MAX_TURNS_PER_USER  : int = 20        # KV deltas kept in memory per user/conversation
MAX_TOKENS_PER_USER : int = 10_000    # Token budget per conversation in memory
MAX_TOTAL_TURNS     : int = 200       # Global pool across all conversations
TTL_SECONDS         : int = 3600      # Evict conversation after this many seconds idle

# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------
EMBEDDER_MODEL : str = "all-MiniLM-L6-v2"   # sentence-transformers model name
EMBEDDING_DIM  : int = 384                   # output dimension of above model

# ---------------------------------------------------------------------------
# Generation defaults
# ---------------------------------------------------------------------------
DEFAULT_MAX_TOKENS  : int   = 200
DEFAULT_TEMPERATURE : float = 1.0
MAX_RETRIES         : int   = 3      # retries if model doesn't emit final channel

# ---------------------------------------------------------------------------
# Chat history (filesystem)
# ---------------------------------------------------------------------------
HISTORY_DIR : str = os.path.join(PROJECT_ROOT, "data", "chat_history")

# ---------------------------------------------------------------------------
# System prompt  (fixed — KV cache pre-computed once at startup)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a helpful, professional AI assistant. "
    "Be concise, accurate, and cite sources when possible. "
    "For every response you must output three channels in this exact order:\n"
    "1. <|channel|>intent<|message|>LABEL — classify the user's intent as exactly one of: "
    "question, creative, technical, emotional, math, explanation, translate, search, none.\n"
    "2. <|channel|>continues<|message|>YES or NO — is this query a continuation of the previous conversation?\n"
    "3. <|channel|>final<|message|>YOUR RESPONSE — your actual answer to the user.\n"
    "Always output all three channels. The first turn of a conversation is always continues=NO."
)

VALID_INTENTS = frozenset({
    "question", "creative", "technical", "emotional",
    "math", "explanation", "translate", "search", "none",
})
