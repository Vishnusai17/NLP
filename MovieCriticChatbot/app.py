"""
Flask Web Server for Movie Critic Chatbot
==========================================
This script serves the LSTM chatbot through a web interface.
"""

from flask import Flask, render_template, request, jsonify, session
from movie_critic_chatbot import MovieCriticChatbot
import os
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Global chatbot instance
chatbot = None


def initialize_chatbot():
    """Initialize or load the chatbot model."""
    global chatbot
    
    if chatbot is None:
        chatbot = MovieCriticChatbot(
            vocab_size=4000,
            embedding_dim=128,
            max_sequence_len=50
        )
        
        # Try to load existing model
        if os.path.exists('movie_critic_model.h5') and os.path.exists('tokenizer.pickle'):
            print("Loading existing model...")
            chatbot.load_model()
            print("Model loaded successfully!")
            return True
        else:
            print("No pre-trained model found.")
            return False
    
    return True


@app.route('/')
def home():
    """Serve the main chat interface."""
    return render_template('index.html')


@app.route('/api/generate', methods=['POST'])
def generate():
    """
    API endpoint to generate text completions.
    
    Expected JSON payload:
    {
        "seed_text": "The acting was",
        "next_words": 30,
        "temperature": 1.0
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'seed_text' not in data:
            return jsonify({
                'success': False,
                'error': 'No seed text provided'
            }), 400
        
        seed_text = data['seed_text'].strip()
        next_words = data.get('next_words', 30)
        temperature = data.get('temperature', 1.0)
        
        if not seed_text:
            return jsonify({
                'success': False,
                'error': 'Seed text cannot be empty'
            }), 400
        
        # Initialize chatbot if needed
        if not initialize_chatbot():
            return jsonify({
                'success': False,
                'error': 'Model not trained yet. Please run movie_critic_chatbot.py first to train the model.'
            }), 500
        
        # Generate text
        generated_text = chatbot.generate_text(
            seed_text,
            next_words=next_words,
            temperature=temperature
        )
        
        return jsonify({
            'success': True,
            'generated_text': generated_text,
            'seed_text': seed_text
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/status', methods=['GET'])
def status():
    """Check if the model is loaded and ready."""
    model_exists = os.path.exists('movie_critic_model.h5') and os.path.exists('tokenizer.pickle')
    model_loaded = chatbot is not None and chatbot.model is not None
    
    return jsonify({
        'model_exists': model_exists,
        'model_loaded': model_loaded,
        'ready': model_loaded or model_exists
    })


if __name__ == '__main__':
    # Initialize chatbot on startup
    initialize_chatbot()
    
    print("\n" + "="*60)
    print("🎬 Movie Critic Chatbot Web Server")
    print("="*60)
    print("\n🌐 Open your browser and go to: http://localhost:5001")
    print("\n" + "="*60 + "\n")
    
    app.run(debug=True, port=5001, host='0.0.0.0')
