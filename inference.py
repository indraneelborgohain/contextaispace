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
from architecture.gptoss20B import Transformer, ModelConfig, RopeScalingConfig

from architecture.gptoss20B import TokenGenerator
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
    generator = TokenGenerator(checkpoint="gpt-oss-20b/original/", device=device)
    prompt = "Once upon a time"
    stop_token_ids = [
    tokenizer.encode("<|end|>")[0],      # 200007
    tokenizer.encode("<|return|>")[0],    # 200002
    tokenizer.encode("<|call|>")[0],      # 200012
    ]
    idx = text_to_token_ids(prompt, tokenizer).to(device)
    prompt = (
    "<|start|>system<|message|>"
    "You are a helpful assistant."
    "<|end|>"
    "<|start|>user<|message|>"
    "What is the capital of France?"
    "<|end|>"
    "<|start|>assistant<|channel|>final<|message|>"
)
    prompt_tokens = tokenizer.encode(prompt)
    stop_token_ids = [200002, 200007, 200012]  # <|return|>, <|end|>, <|call|>

    # Consume the generator and collect all tokens
    output_tokens = list(generator.generate(prompt_tokens, stop_token_ids))
    # Decode tokens to text
    full_output = tokenizer.decode(prompt_tokens + output_tokens)
    print(full_output)
  

