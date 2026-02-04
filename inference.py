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

from transformers import AutoModelForCausalLM
from architecture.gptoss20B import Transformer, ModelConfig, RopeScalingConfig


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
    """Load model from local weights and generate text."""
    
    # Create your custom model instance (empty weights, low memory)
    with torch.device('meta'):
        model = Transformer(ModelConfig)
    
    # Load HuggingFace model with minimal memory footprint
    hf_model = AutoModelForCausalLM.from_pretrained(
        "openai/gpt-oss-20b",
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True)
    
    # Compare architectures (meta device model has no actual weights)
    results = compare_model_architectures(
        model, 
        hf_model, 
        model1_name="Custom GPToss", 
        model2_name="HuggingFace GPT-OSS"
    )
    print_architecture_comparison(results)
    
    # Free memory from both models before loading for inference
    del model
    del hf_model
    import gc
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    local_dir = "model/gpt-oss-20b"
    # Load model from local weights
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = load_gptoss_from_hf(
        local_dir=local_dir,
        device=device,
        strict=True,
    )

    prompt = "Once upon a time"
    output = generate_text(model, prompt, max_tokens=80, temperature=0.8, top_k=50)
    print(output)


if __name__ == "__main__":
    main()

