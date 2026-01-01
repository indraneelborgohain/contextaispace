#!/usr/bin/env python3
"""
Flask server for chat application
Simple backend that returns dummy responses for now
"""
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import time
import random

app = Flask(__name__, static_folder='static')
CORS(app)

# Dummy responses for testing
DUMMY_RESPONSES = [
    "That's a great question! I'm currently running in demo mode with dummy responses.",
    "I understand what you're asking. Once the model is integrated, I'll provide more detailed answers.",
    "Interesting point! This is a placeholder response while we build the real AI backend.",
    "I'm processing your request... Just kidding! This is a demo response for now.",
    "Thanks for your message! The actual model integration is coming soon.",
    "I see what you mean. In the future, I'll be powered by the custom transformer model.",
    "Good question! Right now I'm just a friendly placeholder, but soon I'll have real intelligence!",
    "I appreciate your patience. The real AI is being trained as we speak!",
]


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
        "conversation_id": "optional-conversation-id"
    }
    
    Returns:
    {
        "response": "AI response",
        "timestamp": 1234567890
    }
    """
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        conversation_id = data.get('conversation_id', 'default')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Simulate processing delay
        time.sleep(0.5 + random.random() * 0.5)
        
        # Pick a random dummy response
        response_text = random.choice(DUMMY_RESPONSES)
        
        # Add some context based on message length
        if len(user_message) > 100:
            response_text += " I notice you wrote quite a detailed message!"
        elif '?' in user_message:
            response_text += " That's a good question to ask!"
        
        return jsonify({
            'response': response_text,
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
        'mode': 'demo',
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
    print("🚀 Starting Flask Chat Server")
    print("=" * 60)
    print("📍 Server: http://localhost:5000")
    print("💬 Chat UI: http://localhost:5000")
    print("🔧 API Endpoint: http://localhost:5000/api/chat")
    print("=" * 60)
    print("\nPress CTRL+C to stop the server\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
