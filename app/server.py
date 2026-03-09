#!/usr/bin/env python3
"""
Flask server for GPT-OSS chat application
Provides API endpoints for chat with sentiment-aware system messages
"""
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import time
import torch

from inference import generateResults, generateResultsWithCache, create_models
from system_generator import HybridSystemGenerator
from services.chat_history import get_chat_history_service, ChatHistoryService
from services.kv_cache_store import get_kv_cache_store, KVCacheStore

app = Flask(__name__, static_folder='static')
CORS(app)

# Global variables for models and services
system_gen = None
generator = None
device = None
chat_history: ChatHistoryService = None
kv_cache_store: KVCacheStore = None


def initialize_model():
    """Initialize the model, system generator, chat history, and KV cache at startup."""
    global system_gen, generator, device, chat_history, kv_cache_store
    
    print("Initializing models...")
    
    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize both models using create_models from inference
    generator, system_gen = create_models(device=device)
    
    # Initialize chat history service
    chat_history = get_chat_history_service()
    print(f"Chat history loaded from: {chat_history.history_file}")
    
    # Initialize KV cache store
    kv_cache_store = get_kv_cache_store(max_conversations=100, ttl_seconds=3600)
    print("KV cache store initialized")
    
    print("Model initialization complete!")


@app.route('/')
def index():
    """Serve the main chat interface"""
    return send_from_directory('static', 'index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Chat endpoint that receives user messages and returns responses
    
    Expected JSON:
    {
        "message": "User's question",
        "conversation_id": "optional-conversation-id",
        "max_tokens": 100
    }
    
    Returns:
    {
        "response": "AI response",
        "system_message": "Generated system message",
        "timestamp": 1234567890
    }
    """
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        conversation_id = data.get('conversation_id', 'default')
        max_tokens = data.get('max_tokens', 100)
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Check if we should use KV cache (for continuing conversations)
        use_cache = data.get('use_cache', True)
        
        # Generate system message for debugging/visibility
        system_message = system_gen.generate(user_message) if system_gen else ""
        
        # Save user message to history
        if chat_history:
            chat_history.add_message(conversation_id, "user", user_message)
        
        # Get existing KV cache for this conversation if available
        existing_cache = None
        tokens_so_far = None
        past_turns = None
        if use_cache and kv_cache_store:
            cache_entry = kv_cache_store.get(conversation_id)
            if cache_entry:
                existing_cache = cache_entry.live_cache
                tokens_so_far = cache_entry.tokens_so_far
                past_turns = cache_entry.turns if cache_entry.turns else None
                print(f"Reusing KV cache for conversation {conversation_id} ({len(tokens_so_far)} tokens cached, {len(cache_entry.turns)} turns)")
        
        # Generate response
        start_time = time.time()
        
        if use_cache:
            # Use cache-aware generation with per-turn deltas
            response_text, updated_cache, updated_tokens, turn_delta = generateResultsWithCache(
                user_message,
                generator=generator,
                system_gen=system_gen,
                kv_cache=existing_cache,
                tokens_so_far=tokens_so_far,
                past_turns=past_turns,
                max_tokens=max_tokens
            )
            
            # Store the turn delta and updated cumulative cache
            if kv_cache_store and turn_delta is not None:
                kv_cache_store.add_turn(
                    conversation_id, turn_delta, updated_tokens,
                    live_cache=updated_cache,
                )
        else:
            # Use standard generation (no cache)
            response_text = generateResults(user_message, generator=generator, system_gen=system_gen)
        
        generation_time = time.time() - start_time
        
        # Save assistant response to history
        if chat_history:
            chat_history.add_message(
                conversation_id,
                "assistant",
                response_text,
                system_message=system_message,
                generation_time=round(generation_time, 2)
            )
        
        return jsonify({
            'response': response_text,
            'system_message': system_message,
            'generation_time': round(generation_time, 2),
            'timestamp': int(time.time()),
            'conversation_id': conversation_id,
            'cache_used': use_cache and existing_cache is not None
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'device': str(device) if device else 'not initialized',
        'timestamp': int(time.time())
    })


@app.route('/api/clear', methods=['POST'])
def clear_conversation():
    """Clear conversation history and KV cache for a specific conversation or all."""
    data = request.get_json() or {}
    conversation_id = data.get('conversation_id')
    
    history_cleared = False
    cache_cleared = False
    
    if conversation_id:
        # Clear specific conversation
        if chat_history:
            history_cleared = chat_history.clear_conversation(conversation_id)
        if kv_cache_store:
            cache_cleared = kv_cache_store.clear(conversation_id)
        
        if history_cleared or cache_cleared:
            return jsonify({
                'status': 'cleared',
                'message': f'Conversation {conversation_id} cleared successfully',
                'history_cleared': history_cleared,
                'cache_cleared': cache_cleared
            })
        else:
            return jsonify({
                'status': 'not_found',
                'message': f'Conversation {conversation_id} not found'
            }), 404
    else:
        # Clear all conversations
        history_count = chat_history.clear_all() if chat_history else 0
        cache_count = kv_cache_store.clear_all() if kv_cache_store else 0
        return jsonify({
            'status': 'cleared',
            'message': f'Cleared {history_count} conversation(s) and {cache_count} cache(s)'
        })


@app.route('/api/history', methods=['GET'])
def get_history():
    """Get conversation history."""
    conversation_id = request.args.get('conversation_id')
    
    if not chat_history:
        return jsonify({'error': 'Chat history not initialized'}), 500
    
    if conversation_id:
        # Get specific conversation
        conversation = chat_history.get_conversation(conversation_id)
        if conversation:
            return jsonify(conversation)
        else:
            return jsonify({'error': 'Conversation not found'}), 404
    else:
        # List all conversations
        conversations = chat_history.list_conversations()
        return jsonify({'conversations': conversations})


@app.route('/api/cache/stats', methods=['GET'])
def get_cache_stats():
    """Get KV cache statistics."""
    if not kv_cache_store:
        return jsonify({'error': 'KV cache store not initialized'}), 500
    
    stats = kv_cache_store.get_stats()
    conversations = kv_cache_store.list_conversations()
    
    return jsonify({
        'stats': stats,
        'conversations': conversations
    })


if __name__ == '__main__':
    print("=" * 60)
    print("Starting GPT-OSS Chat Server")
    print("=" * 60)
    
    # Initialize model before starting server
    initialize_model()
    
    print("\n" + "=" * 60)
    print("Server: http://localhost:5000")
    print("Chat UI: http://localhost:5000")
    print("API Endpoint: http://localhost:5000/api/chat")
    print("Health Check: http://localhost:5000/api/health")
    print("=" * 60)
    print("\nPress CTRL+C to stop the server\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
