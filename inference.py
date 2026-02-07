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

# Project root directory (where inference.py is located)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHECKPOINT = os.path.join(PROJECT_ROOT, "model", "gpt-oss-20b", "original")

context_len=4096
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

def generateResults(prompt, generator=None, system_gen=None):
    """
    Generate results for a given prompt.
    
    Args:
        prompt: User's input prompt/query.
        generator: Pre-initialized TokenGenerator. If None, creates a new one.
        system_gen: Pre-initialized HybridSystemGenerator. If None, creates a new one.
    
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
    # Generate
    output_tokens = list(generator.generate(prompt_tokens, stop_token_ids))
    full_output = tokenizer.decode(output_tokens)
    # Extract final answer only (skip reasoning if present)
    if '<|channel|>final<|message|>' in full_output:
        # Has explicit final channel
        final_start = full_output.find('<|channel|>final<|message|>') + len('<|channel|>final<|message|>')
        final_end = full_output.find('<|return|>', final_start)
        if final_end == -1:
            final_end = len(full_output)
        answer = full_output[final_start:final_end].strip()
    else:
        # No channel marker, clean all special tokens
        special_ids = set(tokenizer._special_tokens.values())
        clean_tokens = [t for t in output_tokens if t not in special_ids]
        answer = tokenizer.decode(clean_tokens).strip()
    return answer
    
if __name__ == "__main__":
    prompt = "What is the capital of France?"
    result = generateResults(prompt)
    print(f"Final Answer: {result}")
