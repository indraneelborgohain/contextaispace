import torch
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
from architecture.gptoss20B import Transformer, ModelConfig

from architecture.gptoss20B import TokenGenerator
from system_generator import HybridSystemGenerator, format_prompt_with_system
import os
from typing import List, Tuple, Optional

# Project root directory (where inference.py is located)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHECKPOINT = os.path.join(PROJECT_ROOT, "model", "gpt-oss-20b", "original")

context_len=4096
svd_budget=100  # Number of virtual tokens from SVD compression of overflow KV cache
tokenizer= get_tokenizer()

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

    Strategy:
      - Keep the first (max_len - budget) positions as-is (preserves system prompt
        and early conversation history).
      - Compress the remaining overflow positions into `budget` virtual tokens
        via truncated SVD (top-budget singular components).
      - Final cache size = max_len.

    Args:
        kv_cache: Per-layer list of (K, V) tensors, each (seq_len, ...).
        max_len: Maximum allowed cache length (model context window).
        budget: Number of virtual tokens produced by SVD compression.

    Returns:
        Compressed KV cache with at most max_len positions per layer.
    """
    if kv_cache is None:
        return None
    # Check current length from first layer
    current_len = kv_cache[0][0].shape[0]
    if current_len <= max_len:
        return kv_cache

    keep_len = max_len - budget  # positions preserved verbatim

    compressed = []
    for k, v in kv_cache:
        # k, v shape: (seq_len, num_kv_heads, head_dim) or (seq_len, dim)
        k_keep = k[:keep_len]
        v_keep = v[:keep_len]

        k_overflow = k[keep_len:]  # (overflow_len, ...)
        v_overflow = v[keep_len:]

        k_compressed = _svd_compress_tensor(k_overflow, budget)
        v_compressed = _svd_compress_tensor(v_overflow, budget)

        compressed.append((
            torch.cat([k_keep, k_compressed], dim=0),
            torch.cat([v_keep, v_compressed], dim=0),
        ))

    return compressed


def _svd_compress_tensor(tensor: torch.Tensor, budget: int) -> torch.Tensor:
    """
    Compress sequence positions via truncated SVD.

    Given a tensor of shape (seq_len, ...), flatten to 2-D, compute SVD,
    keep the top `budget` singular components, and reshape back.

    The result has shape (budget, ...) — `budget` virtual token positions
    that capture the most important patterns from the original positions.

    Args:
        tensor: (seq_len, *rest_dims) tensor to compress.
        budget: Number of virtual positions to produce.

    Returns:
        Compressed tensor of shape (budget, *rest_dims).
    """
    orig_shape = tensor.shape  # (seq_len, ...)
    seq_len = orig_shape[0]
    rest_shape = orig_shape[1:]
    hidden = 1
    for d in rest_shape:
        hidden *= d

    # Flatten to 2-D: (seq_len, hidden)
    mat = tensor.reshape(seq_len, hidden).float()  # SVD needs float32

    # Clamp budget to the maximum rank we can extract
    rank = min(budget, seq_len, hidden)

    # Truncated SVD — only compute the top `rank` components
    # U: (seq_len, rank), S: (rank,), Vh: (rank, hidden)
    U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
    # Virtual tokens = diag(S[:rank]) @ Vh[:rank] → (rank, hidden)
    compressed = torch.diag(S[:rank]) @ Vh[:rank]

    # Pad with zeros if rank < budget (e.g. overflow was tiny)
    if rank < budget:
        pad = torch.zeros(
            budget - rank, hidden, dtype=compressed.dtype, device=compressed.device
        )
        compressed = torch.cat([compressed, pad], dim=0)

    # Cast back to original dtype and reshape
    compressed = compressed.to(tensor.dtype).reshape(budget, *rest_shape)
    return compressed

"""
"""
def generateResultsWithCache(
    prompt: str,
    generator,
    system_gen,
    kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    tokens_so_far: Optional[List[int]] = None,
    max_tokens: int = 100,
    temperature: float = 1.0,
    max_retries: int = 3
) -> Tuple[str, List[Tuple[torch.Tensor, torch.Tensor]], List[int]]:

    user_query = prompt

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
        # Continuing conversation: cache already ends with proper <|end|>
        # (we closed it after the previous turn), so just append new user turn
        continuation = (
            f"<|start|>user<|message|>{user_query}<|end|>"
            f"<|start|>assistant"
        )
        new_tokens = tokenizer.encode(continuation, allowed_special='all')

    current_kv_cache = kv_cache
    current_tokens = tokens_so_far if tokens_so_far else []

    output_tokens = None
    updated_kv_cache = None
    all_output_tokens = []  # Accumulate all reasoning steps

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

            # --- FIX 1: Trim <|return|> stop token from cache ---
            clean_cache = _trim_cache(updated_kv_cache, n=1)
            # Compress overflow via SVD to stay within context limit
            clean_cache = _svd_compress_cache(clean_cache, max_len=context_len, budget=svd_budget)

            # --- FIX 2: Properly close the assistant block in the cache ---
            # The model stopped at <|return|> without emitting <|end|>.
            # We feed <|end|> into the model to close the turn cleanly,
            # so the next turn's continuation starts from a valid state.
            closing_tokens = tokenizer.encode("<|end|>", allowed_special='all')
            closing_tensor = torch.as_tensor(
                closing_tokens, dtype=torch.int32, device=generator.device
            )
            with torch.inference_mode():
                _, clean_cache = generator.model(
                    closing_tensor, kv_cache=clean_cache, use_cache=True
                )

            return answer, clean_cache, current_tokens

        print(f"Attempt {attempt + 1}/{max_retries}: No final channel marker, continuing reasoning...")

        # --- REASONING MODEL: Build on previous output ---
        # Compress overflow via SVD to stay within context limit
        current_kv_cache = _svd_compress_cache(updated_kv_cache, max_len=context_len, budget=svd_budget)
        # Next iteration continues from the last output (empty new_tokens since cache has it)
        new_tokens = []

    # Fallback: max retries reached
    print(f"Warning: No final channel marker after {max_retries} attempts, returning cleaned output")
    special_ids = set(tokenizer._special_tokens.values())
    clean_tokens = [t for t in all_output_tokens if t not in special_ids]
    answer = tokenizer.decode(clean_tokens).strip()

    # Still clean up the cache for fallback path so caller isn't stuck with bad state
    if updated_kv_cache is not None:
        clean_cache = _trim_cache(updated_kv_cache, n=1)
        clean_cache = _svd_compress_cache(clean_cache, max_len=context_len, budget=svd_budget)
        closing_tokens = tokenizer.encode("<|end|>", allowed_special='all')
        closing_tensor = torch.as_tensor(
            closing_tokens, dtype=torch.int32, device=generator.device
        )
        with torch.inference_mode():
            _, clean_cache = generator.model(
                closing_tensor, kv_cache=clean_cache, use_cache=True
            )
        return answer, clean_cache, current_tokens

    return answer, updated_kv_cache, current_tokens

if __name__ == "__main__":
    prompt = "What is the capital of France?"
    result = generateResults(prompt)
    print(f"Final Answer: {result}")
