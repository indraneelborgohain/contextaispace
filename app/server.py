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

from inference import generateResults, create_models
from system_generator import HybridSystemGenerator

app = Flask(__name__, static_folder='static')
CORS(app)

# Global variables for models
system_gen = None
generator = None
device = None


def initialize_model():
    """Initialize the model and system generator at startup."""
    global system_gen, generator, device
    
    print("Initializing models...")
    
    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize both models using create_models from inference
    generator, system_gen = create_models(device=device)
    
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
        
        # Generate system message for debugging/visibility
        system_message = system_gen.generate(user_message) if system_gen else ""
        
        # Generate response using generateResults with pre-initialized models
        start_time = time.time()
        response_text = generateResults(user_message, generator=generator, system_gen=system_gen)
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
