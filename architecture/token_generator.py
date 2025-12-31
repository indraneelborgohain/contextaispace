"""
Token Generator for Context-Aware GPT

This module provides the TokenGenerator interface for autoregressive generation
with context management.
"""

import torch


class TokenGenerator:
    """
    Token generation interface for context-aware GPT.
    Automatically manages context across generation steps.
    """
    @torch.inference_mode()
    def __init__(self, checkpoint: str, device: torch.device):
        # Import here to avoid circular dependency
        from .transformer import Transformer
        
        self.device = device
        self.model = Transformer.from_checkpoint(checkpoint, device=self.device)
        self.model.reset_context()

    def reset(self):
        """Reset context for new conversation."""
        self.model.reset_context()

    @torch.inference_mode()
    def generate(self,
                 prompt_tokens: list[int],
                 stop_tokens: list[int],
                 temperature: float = 1.0,
                 max_tokens: int = 0,
                 return_logprobs: bool = False,
                 reset_context: bool = False):
        """
        Generate tokens autoregressively.
        
        Args:
            prompt_tokens: Input token IDs
            stop_tokens: Tokens that signal end of generation
            temperature: Sampling temperature (0.0 = greedy)
            max_tokens: Maximum tokens to generate (0 = no limit)
            return_logprobs: Yield (token, logprob) tuples if True
            reset_context: Reset context before generation if True
        """
        if reset_context:
            self.model.reset_context()
        
        tokens = list(prompt_tokens)
        num_generated_tokens = 0
        
        while max_tokens == 0 or num_generated_tokens < max_tokens:
            logits = self.model(
                torch.as_tensor(tokens, dtype=torch.int32, device=self.device),
                update_context=True
            )[-1]
            
            if temperature == 0.0:
                predicted_token = torch.argmax(logits, dim=-1).item()
            else:
                probs = torch.softmax(logits * (1.0 / temperature), dim=-1)
                predicted_token = torch.multinomial(probs, num_samples=1).item()
            
            tokens.append(predicted_token)
            num_generated_tokens += 1

            if return_logprobs:
                logprobs = torch.log_softmax(logits, dim=-1)
                selected_logprobs = logprobs[predicted_token].item()
                yield predicted_token, selected_logprobs
            else:
                yield predicted_token

            if predicted_token in stop_tokens:
                break
