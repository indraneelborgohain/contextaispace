"""
RunPod Serverless Handler for GPT-OSS 20B Inference

This module implements the RunPod serverless interface for running
GPT-OSS 20B model inference. The model is loaded once during cold start
and reused across requests.
"""

import os
import time
import torch
import runpod

from architecture.tokenizer import get_tokenizer
from architecture.gptoss20B import TokenGenerator
from system_generator import HybridSystemGenerator, format_prompt_with_system

# ── Global state (initialized once per worker) ──────────────────────────

tokenizer = None
generator = None
system_gen = None

MODEL_DIR = os.environ.get("MODEL_DIR", "model/gpt-oss-20b/original/")
HF_REPO_ID = os.environ.get("HF_REPO_ID", "")
HF_LOCAL_DIR = os.environ.get("HF_LOCAL_DIR", "models/gptoss-20b")

# ── Initialization ───────────────────────────────────────────────────────

def load_model():
    """Load the GPT-OSS model and tokenizer once at worker startup."""
    global tokenizer, generator, system_gen

    print("=" * 60)
    print("Initializing GPT-OSS 20B for RunPod Serverless")
    print("=" * 60)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # If a HuggingFace repo is specified, download weights first
    if HF_REPO_ID:
        print(f"Downloading weights from HuggingFace: {HF_REPO_ID}")
        from hf_gptoss_loader import download_hf_weights
        download_hf_weights(repo_id=HF_REPO_ID, local_dir=HF_LOCAL_DIR)
        model_path = HF_LOCAL_DIR
    else:
        model_path = MODEL_DIR

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = get_tokenizer()

    # Load model via TokenGenerator (handles checkpoint loading + KV cache)
    print(f"Loading model from: {model_path}")
    generator = TokenGenerator(checkpoint=model_path, device=device)

    # Load system message generator (runs on CPU)
    print("Loading system message generator...")
    system_gen = HybridSystemGenerator(device=-1)

    print("Model initialization complete!")
    print("=" * 60)


def run_inference(prompt, max_tokens=100, temperature=0.8):
    """
    Run inference on a single prompt.

    Args:
        prompt: The user's input text.
        max_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature.

    Returns:
        Generated text response.
    """
    # Generate system message based on query analysis
    system_message = system_gen.generate(prompt, verbose=False)

    # Format prompt with system message (Harmony format)
    formatted_prompt = format_prompt_with_system(prompt, system_message)

    # Define stop tokens
    stop_token_ids = [
        tokenizer.encode("<|return|>", allowed_special="all")[0],
    ]

    # Tokenize
    prompt_tokens = tokenizer.encode(formatted_prompt, allowed_special="all")

    # Generate tokens
    output_tokens = list(
        generator.generate(
            prompt_tokens,
            stop_token_ids,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    )

    full_output = tokenizer.decode(output_tokens)

    # Extract final answer (skip reasoning if present)
    if "<|channel|>final<|message|>" in full_output:
        final_start = full_output.find("<|channel|>final<|message|>") + len(
            "<|channel|>final<|message|>"
        )
        final_end = full_output.find("<|return|>", final_start)
        if final_end == -1:
            final_end = len(full_output)
        answer = full_output[final_start:final_end].strip()
    else:
        special_ids = set(tokenizer._special_tokens.values())
        clean_tokens = [t for t in output_tokens if t not in special_ids]
        answer = tokenizer.decode(clean_tokens).strip()

    return answer, system_message


# ── RunPod Handler ───────────────────────────────────────────────────────

def handler(job):
    """
    RunPod serverless handler function.

    Expected input:
    {
        "input": {
            "prompt": "What is the capital of France?",
            "max_tokens": 100,        # optional, default 100
            "temperature": 0.8        # optional, default 0.8
        }
    }

    Returns:
    {
        "response": "The generated text...",
        "system_message": "The generated system context...",
        "generation_time": 2.5,
        "tokens_generated": 42
    }
    """
    job_input = job["input"]

    # Validate input
    prompt = job_input.get("prompt", "")
    if not prompt:
        return {"error": "No prompt provided. Include 'prompt' in your input."}

    max_tokens = job_input.get("max_tokens", 100)
    temperature = job_input.get("temperature", 0.8)

    # Clamp parameters
    max_tokens = max(1, min(max_tokens, 4096))
    temperature = max(0.01, min(temperature, 2.0))

    try:
        start_time = time.time()
        answer, system_message = run_inference(prompt, max_tokens, temperature)
        generation_time = time.time() - start_time

        return {
            "response": answer,
            "system_message": system_message,
            "generation_time": round(generation_time, 3),
        }
    except Exception as e:
        return {"error": str(e)}


# ── Entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    load_model()
    runpod.serverless.start({"handler": handler})
