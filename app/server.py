#!/usr/bin/env python3
"""
Flask server for GPT-OSS.

Routes:
    GET  /                      — serve chat UI
    POST /api/chat              — generate a response
    GET  /api/history           — get conversation history
    POST /api/clear             — clear a conversation
    GET  /api/cache/stats       — KV cache memory stats
    GET  /api/health            — health check
"""
import sys
import os
import time

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import inference   # triggers no model load — just imports startup/infer

app = Flask(__name__, static_folder="static")
CORS(app)


def initialize_model():
    """Called once at startup to load the model and warm up caches."""
    print("Initialising GPT-OSS engine...")
    inference.startup()
    print("Engine ready.")



@app.route("/")
def index():
    """Serve the main chat UI."""
    return send_from_directory("static", "index.html")


# ---------------------------------------------------------------------------
# POST /api/chat
# ---------------------------------------------------------------------------

@app.route("/api/chat", methods=["POST"])
def chat():
    """Generate a response.

    Request JSON:
        { "message": str, "conversation_id": str, "max_tokens": int }

    Response JSON:
        { "response": str, "intent": str, "continues": bool,
          "generation_time": float, "timestamp": int, "conversation_id": str }
    """
    try:
        data            = request.get_json() or {}
        user_message    = data.get("message", "").strip()
        conv_id         = data.get("conversation_id", "default")
        max_tokens      = int(data.get("max_tokens", 200))

        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        start = time.time()
        result = inference.infer(
            prompt     = user_message,
            conv_id    = conv_id,
            max_tokens = max_tokens,
        )
        elapsed = round(time.time() - start, 2)

        return jsonify({
            "response":        result["answer"],
            "intent":          result["intent"],
            "continues":       result["continues"],
            "generation_time": elapsed,
            "timestamp":       int(time.time()),
            "conversation_id": conv_id,
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# GET /api/history
# ---------------------------------------------------------------------------

@app.route("/api/history", methods=["GET"])
def get_history():
    """Return chat history.

    Query params:
        conversation_id  — specific conversation (omit for list of all IDs)
    """
    from app.services.chat_history_fs import get_chat_history
    history = get_chat_history()
    conv_id = request.args.get("conversation_id")

    if conv_id:
        messages = history.get_messages(conv_id)
        if not messages:
            return jsonify({"error": "Conversation not found"}), 404
        return jsonify({"conversation_id": conv_id, "messages": messages})

    return jsonify({"conversations": history.list_conversations()})


# ---------------------------------------------------------------------------
# POST /api/clear
# ---------------------------------------------------------------------------

@app.route("/api/clear", methods=["POST"])
def clear_conversation():
    """Clear history and KV cache for a conversation (or all).

    Request JSON:
        { "conversation_id": str }   — omit to clear everything
    """
    from app.services.chat_history_fs import get_chat_history
    from app.services.kv_cache_store import get_kv_cache_store

    data    = request.get_json() or {}
    conv_id = data.get("conversation_id")

    history  = get_chat_history()
    kv_store = get_kv_cache_store()

    if conv_id:
        h = history.clear(conv_id)
        k = kv_store.clear(conv_id)
        if not h and not k:
            return jsonify({"status": "not_found"}), 404
        return jsonify({"status": "cleared", "conversation_id": conv_id})

    h_count = history.clear_all()
    k_count = kv_store.clear_all()
    return jsonify({"status": "cleared", "history": h_count, "kv_caches": k_count})


# ---------------------------------------------------------------------------
# GET /api/cache/stats
# ---------------------------------------------------------------------------

@app.route("/api/cache/stats", methods=["GET"])
def cache_stats():
    """Return KV cache memory usage stats."""
    from app.services.kv_cache_store import get_kv_cache_store
    store = get_kv_cache_store()
    return jsonify({
        "stats":         store.stats(),
        "conversations": store.list_conversations(),
    })


# ---------------------------------------------------------------------------
# GET /api/health
# ---------------------------------------------------------------------------

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "timestamp": int(time.time())})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Starting GPT-OSS Chat Server")
    print("=" * 60)

    initialize_model()

    print("\nServer:  http://localhost:5000")
    print("API:     http://localhost:5000/api/chat")
    print("Health:  http://localhost:5000/api/health")
    print("=" * 60)
    print("\nPress CTRL+C to stop.\n")

    app.run(host="0.0.0.0", port=5000, debug=False)
