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
context_len=4096
tokenizer= get_tokenizer()

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


def main():
    """Load model from saved weights and generate text."""
    
    # Load custom model with transferred HuggingFace weights
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    weights_path = "model/gptoss_custom_weights.pt"
    
    print(f"Loading custom model from '{weights_path}'...")
    model = Transformer(ModelConfig())
    model.from_checkpoint()
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    print(f"✅ Model loaded to {device}")

    prompt = "Once upon a time"
    output = generate_text(model, prompt, max_tokens=80, temperature=0.8, top_k=50)
    print(output)


if __name__ == "__main__":
    device = torch.device("cuda:0")
    
    # Initialize system message generator
    system_gen = HybridSystemGenerator(device=-1)  # Run classifier on CPU
    
    # User query
    user_query = "What is the capital of France?"
    
    # Generate system message based on query sentiment/intent
    system_message = system_gen.generate(user_query, verbose=True)
    print(f"Generated system message: {system_message}\n")
    
    # Format prompt using Harmony format
    prompt = format_prompt_with_system(user_query, system_message)
    
    # Stop tokens
    stop_token_ids = [
        tokenizer.encode("<|end|>", allowed_special='all')[0],      # 200007
        tokenizer.encode("<|return|>", allowed_special='all')[0],   # 200002
        tokenizer.encode("<|call|>", allowed_special='all')[0],     # 200012
    ]
    # Tokenize prompt
    prompt_tokens = tokenizer.encode(prompt, allowed_special='all')
    
    # Generate
    generator = TokenGenerator(checkpoint="model/gpt-oss-20b/original/", device=device)
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
        

    """
    # Extract only the newly generated tokens (exclude the input prompt)
    generated_tokens = output_tokens[len(prompt_tokens):]
    
    # Output format: <|channel|>final<|message|>RESPONSE<|return|>
    # Find <|message|> token in the generated portion and take content after it
    message_token_id = tokenizer.encode("<|message|>", allowed_special='all')[0]  # 200008
    special_token_ids = set(tokenizer._special_tokens.values())
    
    # Find <|message|> in generated tokens - content follows it
    try:
        message_idx = generated_tokens.index(message_token_id)
        response_tokens = generated_tokens[message_idx + 1:]
    except ValueError:
        # No <|message|> found, use all generated tokens
        response_tokens = generated_tokens
    
    # Remove any remaining special tokens (like <|return|>, <|end|>)
    clean_tokens = [t for t in response_tokens if t not in special_token_ids]
    clean_output = tokenizer.decode(clean_tokens)
    """
    
    print(f"Response: {answer}")
  

