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

from architecture.tokenizer import get_tokenizer
from architecture.gptoss20B import TokenGenerator
from system_generator import HybridSystemGenerator, format_prompt_with_system

app = Flask(__name__, static_folder='static')
CORS(app)

# Global variables for model and generator
generator = None
system_gen = None
tokenizer = None
stop_token_ids = None
device = None


def initialize_model():
    """Initialize the model, tokenizer, and system generator."""
    global generator, system_gen, tokenizer, stop_token_ids, device
    
    print("Initializing model...")
    
    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize tokenizer
    tokenizer = get_tokenizer()
    
    # Initialize system message generator (runs on CPU for efficiency)
    system_gen = HybridSystemGenerator(device=-1)
    
    # Stop tokens
    stop_token_ids = [
        tokenizer.encode("<|end|>", allowed_special='all')[0],      # 200007
        tokenizer.encode("<|return|>", allowed_special='all')[0],   # 200002
        tokenizer.encode("<|call|>", allowed_special='all')[0],     # 200012
    ]
    
    # Initialize model
    checkpoint_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "model/gpt-oss-20b/original/"
    )
    
    if os.path.exists(checkpoint_path):
        generator = TokenGenerator(checkpoint=checkpoint_path, device=device)
        print(f"Model loaded from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Running in demo mode without model")
        generator = None
    
    print("Initialization complete!")


def generate_response(user_query: str, max_tokens: int = 100) -> str:
    """
    Generate a response for the user query.
    
    Args:
        user_query: The user's input text
        max_tokens: Maximum tokens to generate
        
    Returns:
        Generated response text
    """
    if generator is None:
        return "Model not loaded. Please ensure the checkpoint is available."
    
    # Generate system message based on query sentiment/intent
    system_message = system_gen.generate(user_query)
    
    # Format prompt
    prompt = format_prompt_with_system(user_query, system_message)
    
    # Tokenize
    prompt_tokens = tokenizer.encode(prompt, allowed_special='all')
    
    # Generate
    output_tokens = list(generator.generate(
        prompt_tokens, 
        stop_token_ids,
        max_tokens=max_tokens
    ))
    
    # Clean decode (remove special tokens)
    special_token_ids_set = set(tokenizer._special_tokens.values())
    clean_tokens = [t for t in output_tokens if t not in special_token_ids_set]
    
    return tokenizer.decode(clean_tokens)


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
        
        # Generate system message for debugging/visibility
        system_message = system_gen.generate(user_message) if system_gen else ""
        
        # Generate response
        start_time = time.time()
        response_text = generate_response(user_message, max_tokens=max_tokens)
        generation_time = time.time() - start_time
        
        return jsonify({
            'response': response_text,
            'system_message': system_message,
            'generation_time': round(generation_time, 2),
            'timestamp': int(time.time()),
            'conversation_id': conversation_id
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'mode': 'model' if generator is not None else 'demo',
        'device': str(device) if device else 'not initialized',
        'timestamp': int(time.time())
    })


@app.route('/api/clear', methods=['POST'])
def clear_conversation():
    """Clear conversation history (placeholder for now)"""
    return jsonify({
        'status': 'cleared',
        'message': 'Conversation cleared successfully'
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
