"""
Flask Web Application for Intent Classifier
Provides a web UI and API for intent classification
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import json
import pickle
import numpy as np
import random
from tensorflow import keras
import os

app = Flask(__name__)
CORS(app)

# Global variables for model artifacts
model = None
vectorizer = None
label_encoder = None
responses = None


def load_artifacts():
    """Load all model artifacts on startup."""
    global model, vectorizer, label_encoder, responses
    
    print("Loading model artifacts...")
    model = keras.models.load_model('intent_classifier_model.h5')
    
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    with open('responses.json', 'r') as f:
        responses = json.load(f)
    
    print("Model loaded successfully!")


def predict_intent(text):
    """
    Predict intent from user text.
    
    Args:
        text (str): User input text
    
    Returns:
        dict: Prediction results including intent, confidence, and all probabilities
    """
    # Preprocess
    text_lower = text.lower()
    text_vector = vectorizer.transform([text_lower]).toarray()
    
    # Predict
    predictions = model.predict(text_vector, verbose=0)[0]
    
    # Get results
    predicted_class = np.argmax(predictions)
    predicted_intent = label_encoder.classes_[predicted_class]
    confidence = float(predictions[predicted_class])
    
    # All probabilities
    all_probs = {
        label_encoder.classes_[i]: float(predictions[i])
        for i in range(len(predictions))
    }
    
    # Get response
    response_text = random.choice(responses[predicted_intent]) if predicted_intent in responses else \
        "I'm not sure how to help with that. Please contact us at (555) 123-4567."
    
    return {
        'intent': predicted_intent,
        'confidence': confidence,
        'response': response_text,
        'all_probabilities': all_probs
    }


@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.route('/static/<path:path>')
def send_static(path):
    """Serve static files."""
    return send_from_directory('static', path)


@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for intent prediction.
    
    Expects JSON: {"text": "user input"}
    Returns JSON with prediction results
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        user_text = data['text'].strip()
        
        if not user_text:
            return jsonify({'error': 'Empty text provided'}), 400
        
        # Get prediction
        result = predict_intent(user_text)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})


if __name__ == '__main__':
    # Load model on startup
    load_artifacts()
    
    print("\n" + "="*50)
    print("INTENT CLASSIFIER WEB APP")
    print("="*50)
    print("\nServer starting on http://localhost:5001")
    print("Open your browser to interact with the bot!\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001)
