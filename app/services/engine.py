"""
Inference engine — all generation logic lives here.

Public surface:
    startup(device, checkpoint) → None        initialise model + system cache (call once)
    infer(prompt, conv_id, **kwargs) → dict   the only call inference.py needs to make

Everything else is private to this module.
"""

import os
import re
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional

from app.config import (
    DEFAULT_CHECKPOINT,
    CONTEXT_LIMIT,
    LAST_N_BUDGET,
    TOP_K_TURNS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    SYSTEM_PROMPT,
    VALID_INTENTS,
)

# Layer index that always keeps full context (odd layers).
# Used to read true cache length — even layers use sliding window.
_FULL_CTX_LAYER: int = 1

# ---------------------------------------------------------------------------
# Module-level singletons — populated by startup()
# ---------------------------------------------------------------------------
_generator        = None   # TokenGenerator
_system_kv_cache  = None   # List[Tuple[Tensor, Tensor]]  — frozen, never mutated
_tokenizer        = None   # tiktoken Encoding


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def startup(device=None, checkpoint: str = None) -> None:
    """Load model and pre-compute system-prompt KV cache.

    Call once at application startup before serving any requests.
    """
    global _generator, _system_kv_cache, _tokenizer

    from architecture.tokenizer import get_tokenizer
    from architecture.gptoss20B import TokenGenerator

    if checkpoint is None:
        checkpoint = DEFAULT_CHECKPOINT
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    _tokenizer = get_tokenizer()
    _generator = TokenGenerator(checkpoint=checkpoint, device=device)
    _system_kv_cache = _cache_system_prompt()
    print(f"[engine] Model loaded on {device}. System prompt cached "
          f"({_system_kv_cache[_FULL_CTX_LAYER][0].shape[0]} tokens).")


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def infer(
    prompt: str,
    conv_id: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> dict:
    """Generate a response for *prompt* within conversation *conv_id*.

    Builds a flat token sequence:
        [system prompt] + [retrieved past turns as text] + [user message]
    and passes it directly to the model in a single forward pass.
    No KV cache pre-loading is used.

    Args:
        prompt:      Plain-text user message.
        conv_id:     Conversation identifier (any unique string per session).
        max_tokens:  Maximum tokens to generate.
        temperature: Sampling temperature.

    Returns:
        {
            "answer":    str,   # the model's final response text
            "intent":    str,   # classified intent label
            "continues": bool,  # whether this turn depends on prior context
        }
    """
    _assert_ready()

    from app.services.kv_cache_store import get_kv_cache_store, TurnDelta
    from app.services.embedder import get_embedder
    from app.services.chat_history_fs import get_chat_history

    kv_store = get_kv_cache_store()
    embedder = get_embedder()
    history  = get_chat_history()

    # --- Retrieve past turns for this conversation ---
    past_turns = kv_store.get_turns(conv_id)

    # --- Build full prompt token sequence ---
    # System prompt first
    system_str  = f"<|start|>system<|message|>{SYSTEM_PROMPT}<|end|>"
    prompt_tokens = _tokenizer.encode(system_str, allowed_special="all")

    # Select relevant past turns to fill remaining token budget
    turn_budget = CONTEXT_LIMIT - len(prompt_tokens) - LAST_N_BUDGET
    if past_turns and turn_budget > 0:
        context_turns = _select_text_turns(past_turns, prompt, turn_budget)
        for turn in context_turns:
            prompt_tokens += turn.token_ids

    # Append the current user message
    user_str    = f"<|start|>user<|message|>{prompt}<|end|><|start|>assistant<|channel|>"
    user_tokens = _tokenizer.encode(user_str, allowed_special="all")
    prompt_tokens += user_tokens

    # Classify intent and continuation locally — no model output needed
    intent    = _classify_intent(prompt)
    continues = bool(past_turns)

    # --- Generate ---
    answer, output_tokens = _generate_loop(
        prompt_tokens = prompt_tokens,
        max_tokens    = max_tokens,
        temperature   = temperature,
    )

    # --- Build and store TurnDelta (token_ids + embedding only; no KV delta) ---
    turn_tokens = user_tokens + output_tokens
    turn_text   = _tokenizer.decode(turn_tokens)
    embedding   = embedder.encode(turn_text)

    turn_delta = TurnDelta(
        kv_delta   = [],           # not used in full-text mode
        token_ids  = turn_tokens,
        embedding  = embedding,
        start_pos  = 0,            # not used in full-text mode
        num_tokens = len(turn_tokens),
        intent     = intent,
        continues  = continues,
    )
    kv_store.add_turn(conv_id, turn_delta)

    # --- Persist to chat history ---
    history.add_turn(conv_id, role="user",      content=prompt)
    history.add_turn(conv_id, role="assistant", content=answer,
                     intent=intent, continues=continues)

    return {"answer": answer, "intent": intent, "continues": continues}


# ---------------------------------------------------------------------------
# Full-text context turn selection
# ---------------------------------------------------------------------------

def _select_text_turns(past_turns: list, query_text: str, token_budget: int) -> list:
    """Return a chronologically-ordered subset of past_turns whose combined
    token_ids length fits within token_budget.

    Strategy (mirrors _assemble_kv_cache):
      1. Walk backwards from the most recent turn, pinning turns until
         LAST_N_BUDGET tokens are consumed.
      2. Fill any remaining budget with the top-K semantically similar
         turns from the rest.
    """
    recent_turns  = []
    recent_tokens = 0
    remaining     = list(past_turns)

    while remaining and recent_tokens < LAST_N_BUDGET:
        turn = remaining.pop()                 # most recent first
        recent_turns.insert(0, turn)
        recent_tokens += turn.num_tokens

    middle_budget = token_budget - recent_tokens
    selected_middle = []

    if middle_budget > 0 and remaining:
        from app.services.lsi_retrieval import select_turns
        indices = select_turns(
            turns        = remaining,
            query_text   = query_text,
            token_budget = middle_budget,
            top_k        = TOP_K_TURNS,
        )
        selected_middle = [remaining[i] for i in indices]

    return selected_middle + recent_turns


# ---------------------------------------------------------------------------
# KV cache assembly (kept for reference — not used in full-text mode)
# ---------------------------------------------------------------------------

def _assemble_kv_cache(
    past_turns: list,
    query_text: str,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Build the KV cache to use for the current turn.

    Layout: [system_cache] + [selected past turn deltas, re-RoPE'd]

    If total tokens fit within CONTEXT_LIMIT → include all turns.
    Otherwise:
      - Walk backwards from the most recent turn, greedily filling up to
        LAST_N_BUDGET tokens.  This ensures the full 1000-token budget is
        always consumed by recent turns even when individual turns are short.
      - Fill any remaining context budget with top-K semantically similar
        turns from the history that was not already included above.
    """
    from architecture.gptoss20B import reposition_rope

    # Clone system cache — never mutate the frozen singleton
    if not past_turns:
        return [(k.clone(), v.clone()) for k, v in _system_kv_cache]

    system_len = _system_kv_cache[_FULL_CTX_LAYER][0].shape[0]
    total_turn_tokens = sum(t.num_tokens for t in past_turns)

    # --- Case 1: everything fits ---
    if system_len + total_turn_tokens <= CONTEXT_LIMIT:
        return _concat_turns(_system_kv_cache, past_turns)

    # --- Case 2: overflow ---
    # Step A: walk backwards, accumulating recent turns until LAST_N_BUDGET
    # is fully consumed (or there are no more turns).
    recent_turns   = []
    recent_tokens  = 0
    remaining_past = list(past_turns)          # copy so we can pop

    while remaining_past and recent_tokens < LAST_N_BUDGET:
        turn = remaining_past.pop()            # take from the end (most recent)
        recent_turns.insert(0, turn)           # preserve chronological order
        recent_tokens += turn.num_tokens

    # Step B: fill leftover context with semantically similar middle turns.
    middle_budget = CONTEXT_LIMIT - system_len - recent_tokens

    if middle_budget > 0 and remaining_past:
        from app.services.lsi_retrieval import select_turns
        selected_indices = select_turns(
            turns        = remaining_past,
            query_text   = query_text,
            token_budget = middle_budget,
            top_k        = TOP_K_TURNS,
        )
        selected_middle = [remaining_past[i] for i in selected_indices]
    else:
        selected_middle = []

    turns_to_use = selected_middle + recent_turns
    return _concat_turns(_system_kv_cache, turns_to_use)


def _concat_turns(
    system_cache: List[Tuple[torch.Tensor, torch.Tensor]],
    turns: list,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Concatenate system cache + turn deltas, re-RoPE'ing each delta."""
    from architecture.gptoss20B import reposition_rope

    num_layers   = len(system_cache)
    system_len   = system_cache[_FULL_CTX_LAYER][0].shape[0]
    rope_modules = [block.attn.rope for block in _generator.model.block]
    dev          = _generator.device

    assembled = []
    for layer_idx in range(num_layers):
        parts_k = [system_cache[layer_idx][0].clone()]
        parts_v = [system_cache[layer_idx][1].clone()]
        pos_offset = system_len

        for turn in turns:
            delta = turn.kv_delta[layer_idx] if turn.kv_loaded else None
            if delta is None:
                continue                        # sliding-window or evicted layer
            k_delta, v_delta = delta
            k_delta = k_delta.to(dev)
            v_delta = v_delta.to(dev)

            old_pos = torch.arange(
                turn.start_pos, turn.start_pos + turn.num_tokens, device=dev
            )
            new_pos = torch.arange(
                pos_offset, pos_offset + turn.num_tokens, device=dev
            )
            k_repositioned = reposition_rope(
                k_delta, old_pos, new_pos, rope_modules[layer_idx]
            )
            parts_k.append(k_repositioned)
            parts_v.append(v_delta)
            pos_offset += turn.num_tokens

        assembled.append((
            torch.cat(parts_k, dim=0),
            torch.cat(parts_v, dim=0),
        ))
    return assembled


# ---------------------------------------------------------------------------
# Generation loop
# ---------------------------------------------------------------------------

def _generate_loop(
    prompt_tokens: List[int],
    max_tokens: int,
    temperature: float,
) -> Tuple[str, List[int]]:
    """Run the model on a flat token sequence and return when the stop token
    (<|return|>) is emitted or max_tokens is reached.

    Returns:
        (answer, output_tokens)
    """
    stop_ids = [_tokenizer.encode("<|return|>", allowed_special="all")[0]]

    output_tokens = list(_generator.generate(
        prompt_tokens = prompt_tokens,
        stop_tokens   = stop_ids,
        temperature   = temperature,
        max_tokens    = max_tokens,
    ))

    full_output = _tokenizer.decode(output_tokens)
    answer      = _extract_final(full_output)
    return answer, output_tokens


# ---------------------------------------------------------------------------
# System prompt caching
# ---------------------------------------------------------------------------

def _cache_system_prompt() -> List[Tuple[torch.Tensor, torch.Tensor]]:
    prompt_str = f"<|start|>system<|message|>{SYSTEM_PROMPT}<|end|>"
    tokens     = _tokenizer.encode(prompt_str, allowed_special="all")
    tensor     = torch.as_tensor(tokens, dtype=torch.int32, device=_generator.device)
    with torch.inference_mode():
        _, kv = _generator.model(tensor, kv_cache=None, use_cache=True)
    return kv


# ---------------------------------------------------------------------------
# KV cache helpers
# ---------------------------------------------------------------------------

def _trim_cache(
    kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
    n: int = 1,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Remove last *n* token positions from every layer."""
    return [(k[:-n], v[:-n]) for k, v in kv_cache]


def _extract_delta(
    kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
    start: int,
    end: int,
) -> list:
    """Slice positions [start:end] from cumulative KV cache per layer.

    Returns None for sliding-window layers whose cache is shorter than *end*.
    """
    deltas = []
    for k, v in kv_cache:
        if k.shape[0] < end or start >= end:
            deltas.append(None)
        else:
            deltas.append((k[start:end].clone(), v[start:end].clone()))
    return deltas


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def _classify_intent(prompt: str) -> str:
    """Classify user intent from the prompt text using keyword heuristics."""
    t = prompt.lower()
    if any(w in t for w in ["translate", "in french", "in spanish", "in german", "in japanese"]):
        return "translate"
    if any(w in t for w in ["calculate", "solve", "equation", "integral", "derivative", "factorial"]):
        return "math"
    if any(w in t for w in ["code", "function", "bug", "debug", "implement", "python", "javascript", "error"]):
        return "technical"
    if any(w in t for w in ["write a story", "write a poem", "creative", "imagine", "fiction"]):
        return "creative"
    if any(w in t for w in ["feel", "sad", "happy", "anxious", "depressed", "worried", "emotion"]):
        return "emotional"
    if any(w in t for w in ["search for", "look up", "find me", "latest news"]):
        return "search"
    if any(w in t for w in ["explain", "how does", "what is", "describe", "definition of"]):
        return "explanation"
    if "?" in prompt:
        return "question"
    return "none"


def _extract_final(text: str) -> str:
    """Extract the user-facing answer from model output.

    Tries Harmony channel markers in order of precedence:
        <|channel|>final<|message|>ANSWER        (no # prefix)
        <|channel|>#final<|message|>ANSWER       (# prefix variant)
    Falls back to stripping all special tokens from the full output.
    """
    start = -1
    marker_len = 0
    for marker in ("<|channel|>final<|message|>", "<|channel|>#final<|message|>"):
        idx = text.find(marker)
        if idx != -1:
            start = idx + len(marker)
            marker_len = len(marker)
            break

    if start != -1:
        # Answer ends at the next channel marker, return token, or end of text
        for stopper in ("<|channel|>", "<|return|>", "<|end|>"):
            idx = text.find(stopper, start)
            if idx != -1:
                return text[start:idx].strip()
        return text[start:].strip()

    # Fallback: remove all <|...|> special tokens and return cleaned text
    return re.sub(r"<\|[^|>]*\|>", "", text).strip()


def _strip_special(token_ids: List[int]) -> str:
    special_ids = set(_tokenizer._special_tokens.values())
    clean       = [t for t in token_ids if t not in special_ids]
    return _tokenizer.decode(clean).strip()


# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------

def _assert_ready() -> None:
    if _generator is None or _system_kv_cache is None:
        raise RuntimeError(
            "Engine not initialised. Call engine.startup() before engine.infer()."
        )
