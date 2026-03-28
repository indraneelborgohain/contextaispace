"""
inference.py — stateless public entry point.

This file contains NO logic. It is a thin wrapper that delegates
everything to app.services.engine.

External callers (server, RunPod handler, scripts) should only import
from here:

    from inference import startup, infer
"""

from app.services.engine import startup, infer

__all__ = ["startup", "infer"]


if __name__ == "__main__":
    # Quick smoke-test: initialise and run one turn
    startup()
    result = infer(
        prompt  = "What is the capital of France?",
        conv_id = "smoke-test",
    )
    print(f"Intent:    {result['intent']}")
    print(f"Continues: {result['continues']}")
    print(f"Answer:    {result['answer']}")
