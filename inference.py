import torch
import numpy as np
from torch.nn import functional as F

from architecture.tokenizer import get_tokenizer
from hf_gptoss_loader import load_gptoss_from_hf
from architecture.model_loader import (
    compare_model_architectures,
    print_architecture_comparison,
    copy_weights,
    print_copy_results,
    create_weight_mapping,
)

#from transformers import AutoModelForCausalLM
from architecture.gptoss20B import Transformer, ModelConfig, reposition_rope

from architecture.gptoss20B import TokenGenerator
from system_generator import HybridSystemGenerator, format_prompt_with_system
import os
from typing import List, Tuple, Optional

# Project root directory (where inference.py is located)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHECKPOINT = os.path.join(PROJECT_ROOT, "model", "gpt-oss-20b", "original")

# ---------------------------------------------------------------------------
# Context-window budget layout  (configurable)
#     context_len  = prefix_budget + lsi_budget + current_budget
#     4096         = 3096          + 500        + 500
# ---------------------------------------------------------------------------
context_len: int = 4096
prefix_budget: int = 3096   # Most-recent KV kept verbatim
lsi_budget: int = 500       # Filled by LSI-retrieved past turn deltas
current_budget: int = 500   # Reserved for the current turn

# Legacy SVD budget kept as fallback (unused in LSI path)
svd_budget: int = 100

tokenizer = get_tokenizer()

def create_models(device=None, checkpoint=None):
    """
    Initialize and return the model and system generator for reuse.
    
    Args:
        device: torch device to use. Defaults to cuda:0 if available.
        checkpoint: Path to model checkpoint. Defaults to model/gpt-oss-20b/original/ in project root.
    
    Returns:
        tuple: (TokenGenerator, HybridSystemGenerator)
    """
    if checkpoint is None:
        checkpoint = DEFAULT_CHECKPOINT
    
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Initialize system message generator (runs on CPU for efficiency)
    system_gen = HybridSystemGenerator(device=-1)
    
    # Initialize token generator with the model
    generator = TokenGenerator(checkpoint=checkpoint, device=device)
   
    
    return generator, system_gen

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    return tokenizer.decode(token_ids.tolist())

def generate_text(model, prompt, max_tokens=100, temperature=0.8, top_k=50):
    """Generate text from a prompt using trained model."""
    device = next(model.parameters()).device
    model.eval()
    
    # Tokenize input
    idx = text_to_token_ids(prompt, tokenizer).to(device)
    
    # Generate
    for _ in range(max_tokens):
        idx_cond = idx[-context_len:]
        with torch.inference_mode():
            # Model expects (B, T) input and returns (logits, aux_dict)
            logits, _ = model(idx_cond.unsqueeze(0))  # add batch dim
        # logits shape: (1, T, vocab_size) -> take last token
        logits = logits[0, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[[-1]]] = -float('Inf')

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=0)
    # Decode and return
    result = token_ids_to_text(idx,tokenizer)
    return result

def generateResults(prompt, generator=None, system_gen=None, max_retries: int = 3):
    """
    Generate results for a given prompt.
    
    Args:
        prompt: User's input prompt/query.
        generator: Pre-initialized TokenGenerator. If None, creates a new one.
        system_gen: Pre-initialized HybridSystemGenerator. If None, creates a new one.
        max_retries: Maximum number of retries if final channel marker is missing.
    
    Returns:
        str: Generated answer text.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Use provided models or create new ones
    if generator is None or system_gen is None:
        generator, system_gen = create_models(device=device)
    
    # User query
    user_query = prompt
    # Generate system message based on query sentiment/intent
    system_message = system_gen.generate(user_query, verbose=True)
    print(f"Generated system message: {system_message}\n")
    # Format prompt using Harmony format
    prompt = format_prompt_with_system(user_query, system_message)
    # Stop tokens
    stop_token_ids = [
    tokenizer.encode("<|return|>", allowed_special='all')[0], # 200002
    ]
    # Tokenize prompt
    prompt_tokens = tokenizer.encode(prompt, allowed_special='all')
    
    # Generate with retry until final channel marker is present
    for attempt in range(max_retries):
        output_tokens = list(generator.generate(prompt_tokens, stop_token_ids))
        full_output = tokenizer.decode(output_tokens)
        
        if '<|channel|>final<|message|>' in full_output:
            # Has explicit final channel - extract answer
            final_start = full_output.find('<|channel|>final<|message|>') + len('<|channel|>final<|message|>')
            final_end = full_output.find('<|return|>', final_start)
            if final_end == -1:
                final_end = len(full_output)
            answer = full_output[final_start:final_end].strip()
            return answer
        
        # No final channel marker, retry
        print(f"Attempt {attempt + 1}/{max_retries}: No final channel marker, retrying...")
    
    # Max retries reached, return cleaned output as fallback
    print(f"Warning: No final channel marker after {max_retries} attempts, returning cleaned output")
    special_ids = set(tokenizer._special_tokens.values())
    clean_tokens = [t for t in output_tokens if t not in special_ids]
    answer = tokenizer.decode(clean_tokens).strip()
    return answer

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
    current_len = kv_cache[0][0].shape[0]
    if current_len <= max_len:
        return kv_cache

    keep_len = max_len - budget

    compressed = []
    for k, v in kv_cache:
        k_keep = k[:keep_len]
        v_keep = v[:keep_len]

        k_compressed = _svd_compress_tensor(k[keep_len:], budget)
        v_compressed = _svd_compress_tensor(v[keep_len:], budget)

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
    for k, v in kv_cache:
        if k.shape[0] < end or start >= end:
            # Sliding-window layer or empty range
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

    current_len = recent_kv[0][0].shape[0]

    # If still within context window, nothing to do
    if current_len <= context_len:
        return recent_kv

    # --- A. Keep the first prefix_len positions (system prompt + early context) ---
    actual_prefix = min(prefix_len, current_len)
    prefix_slice = [(k[:actual_prefix], v[:actual_prefix]) for k, v in recent_kv]

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

"""
"""
def generateResultsWithCache(
    prompt: str,
    generator,
    system_gen,
    kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    tokens_so_far: Optional[List[int]] = None,
    past_turns: Optional[list] = None,     # List[TurnDelta] from KVCacheStore
    max_tokens: int = 100,
    temperature: float = 1.0,
    max_retries: int = 3
) -> tuple:
    """Generate a response with KV-cache continuation and per-turn delta tracking.

    Returns:
        (answer, clean_cache, all_tokens, turn_delta)
        * clean_cache  — the KV cache for the *current* turn only (delta).
        * turn_delta   — a TurnDelta ready to be stored in KVCacheStore.
    """
    from app.services.kv_cache_store import TurnDelta

    user_query = prompt
    vocab_size = ModelConfig.vocab_size

    stop_token_ids = [
        tokenizer.encode("<|return|>", allowed_special='all')[0],
    ]

    if kv_cache is None or tokens_so_far is None:
        # New conversation: format full prompt with system message
        system_message = system_gen.generate(user_query, verbose=True)
        print(f"Generated system message: {system_message}\n")
        formatted_prompt = format_prompt_with_system(user_query, system_message)
        new_tokens = tokenizer.encode(formatted_prompt, allowed_special='all')
        tokens_so_far = []
    else:
        continuation = (
            f"<|start|>user<|message|>{user_query}<|end|>"
            f"<|start|>assistant"
        )
        new_tokens = tokenizer.encode(continuation, allowed_special='all')

    # Track where this turn's KV delta starts (position in cumulative cache)
    turn_kv_start = kv_cache[0][0].shape[0] if kv_cache is not None else 0

    current_kv_cache = kv_cache
    current_tokens = tokens_so_far if tokens_so_far else []

    output_tokens = None
    updated_kv_cache = None
    all_output_tokens = []

    for attempt in range(max_retries):
        output_tokens, updated_kv_cache = generator.generate_with_cache(
            new_tokens=new_tokens,
            stop_tokens=stop_token_ids,
            kv_cache=current_kv_cache,
            temperature=temperature,
            max_tokens=max_tokens
        )

        all_output_tokens.extend(output_tokens)
        current_tokens = current_tokens + new_tokens + output_tokens
        full_output = tokenizer.decode(all_output_tokens)

        if '<|channel|>final<|message|>' in full_output:
            final_start = full_output.find('<|channel|>final<|message|>') + len('<|channel|>final<|message|>')
            final_end = full_output.find('<|return|>', final_start)
            if final_end == -1:
                final_end = len(full_output)
            answer = full_output[final_start:final_end].strip()

            # Trim <|return|> stop token from cache
            clean_cache = _trim_cache(updated_kv_cache, n=1)

            # --- Extract this turn's KV delta BEFORE compression ------
            turn_kv_end = clean_cache[0][0].shape[0]
            turn_token_ids = new_tokens + all_output_tokens
            turn_delta = TurnDelta(
                kv_delta=_extract_turn_delta(clean_cache, turn_kv_start, turn_kv_end),
                token_ids=turn_token_ids,
                bow_vector=_build_bow(turn_token_ids, vocab_size),
                start_pos=turn_kv_start,
                num_tokens=turn_kv_end - turn_kv_start,
            )

            # --- Compress via LSI if over context limit ---------------
            all_past = (past_turns or []) + [turn_delta]
            clean_cache = assemble_cache_with_lsi(
                recent_kv=clean_cache,
                past_turns=all_past,
                query_token_ids=new_tokens,
                generator=generator,
            )

            # Close the assistant block in the cache
            closing_tokens = tokenizer.encode("<|end|>", allowed_special='all')
            closing_tensor = torch.as_tensor(
                closing_tokens, dtype=torch.int32, device=generator.device
            )
            with torch.inference_mode():
                _, clean_cache = generator.model(
                    closing_tensor, kv_cache=clean_cache, use_cache=True
                )

            return answer, clean_cache, current_tokens, turn_delta

        print(f"Attempt {attempt + 1}/{max_retries}: No final channel marker, continuing reasoning...")

        # Reasoning continuation: compress if needed, keep going
        current_kv_cache = assemble_cache_with_lsi(
            recent_kv=updated_kv_cache,
            past_turns=past_turns or [],
            query_token_ids=new_tokens,
            generator=generator,
        )
        new_tokens = []

    # Fallback: max retries reached
    print(f"Warning: No final channel marker after {max_retries} attempts, returning cleaned output")
    special_ids = set(tokenizer._special_tokens.values())
    clean_tokens = [t for t in all_output_tokens if t not in special_ids]
    answer = tokenizer.decode(clean_tokens).strip()

    # Build turn delta even for fallback
    turn_token_ids = new_tokens + all_output_tokens
    turn_delta = None

    if updated_kv_cache is not None:
        clean_cache = _trim_cache(updated_kv_cache, n=1)

        turn_kv_end = clean_cache[0][0].shape[0]
        turn_delta = TurnDelta(
            kv_delta=_extract_turn_delta(clean_cache, turn_kv_start, turn_kv_end),
            token_ids=turn_token_ids,
            bow_vector=_build_bow(turn_token_ids, vocab_size),
            start_pos=turn_kv_start,
            num_tokens=turn_kv_end - turn_kv_start,
        )

        all_past = (past_turns or []) + ([turn_delta] if turn_delta else [])
        clean_cache = assemble_cache_with_lsi(
            recent_kv=clean_cache,
            past_turns=all_past,
            query_token_ids=turn_token_ids,
            generator=generator,
        )

        closing_tokens = tokenizer.encode("<|end|>", allowed_special='all')
        closing_tensor = torch.as_tensor(
            closing_tokens, dtype=torch.int32, device=generator.device
        )
        with torch.inference_mode():
            _, clean_cache = generator.model(
                closing_tensor, kv_cache=clean_cache, use_cache=True
            )
        return answer, clean_cache, current_tokens, turn_delta

    return answer, updated_kv_cache, current_tokens, turn_delta

if __name__ == "__main__":
    prompt = "What is the capital of France?"
    result = generateResults(prompt)
    print(f"Final Answer: {result}")
