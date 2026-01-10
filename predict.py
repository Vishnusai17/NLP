"""
Intent Prediction Script
Loads trained model and classifies user input
"""

import json
import pickle
import numpy as np
import random
import sys
from tensorflow import keras


def load_model_artifacts():
    """
    Load the trained model, vectorizer, and label encoder.
    
    Returns:
        tuple: (model, vectorizer, label_encoder, responses)
    """
    # Load model
    model = keras.models.load_model('intent_classifier_model.h5')
    
    # Load vectorizer
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    # Load label encoder
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    # Load responses
    with open('responses.json', 'r') as f:
        responses = json.load(f)
    
    return model, vectorizer, label_encoder, responses


def predict_intent(text, model, vectorizer, label_encoder):
    """
    Predict the intent of user input text.
    
    Args:
        text (str): User input text
        model (keras.Model): Trained model
        vectorizer (TfidfVectorizer): Fitted vectorizer
        label_encoder (LabelEncoder): Fitted label encoder
    
    Returns:
        tuple: (predicted_intent, confidence, all_probabilities)
    """
    # Preprocess input
    text_lower = text.lower()
    text_vector = vectorizer.transform([text_lower]).toarray()
    
    # Predict
    predictions = model.predict(text_vector, verbose=0)[0]
    
    # Get predicted class
    predicted_class = np.argmax(predictions)
    predicted_intent = label_encoder.classes_[predicted_class]
    confidence = predictions[predicted_class]
    
    # Get all probabilities
    all_probs = {
        label_encoder.classes_[i]: float(predictions[i])
        for i in range(len(predictions))
    }
    
    return predicted_intent, confidence, all_probs


def get_response(intent, responses):
    """
    Get a random response for the predicted intent.
    
    Args:
        intent (str): Predicted intent
        responses (dict): Dictionary of intent responses
    
    Returns:
        str: Response text
    """
    if intent in responses:
        return random.choice(responses[intent])
    else:
        return "I'm not sure how to help with that. Please contact us at (555) 123-4567."


def main():
    """Main function for command-line usage."""
    print("Loading model...")
    model, vectorizer, label_encoder, responses = load_model_artifacts()
    print("Model loaded successfully!\n")
    
    if len(sys.argv) > 1:
        # Command-line mode with argument
        user_input = ' '.join(sys.argv[1:])
        intent, confidence, all_probs = predict_intent(user_input, model, vectorizer, label_encoder)
        response = get_response(intent, responses)
        
        print(f"User: {user_input}")
        print(f"\nPredicted Intent: {intent}")
        print(f"Confidence: {confidence*100:.2f}%")
        print(f"\nAll Probabilities:")
        for intent_name, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
            print(f"  {intent_name}: {prob*100:.2f}%")
        print(f"\nBot Response: {response}")
    else:
        # Interactive mode
        print("="*50)
        print("INTENT CLASSIFIER - INTERACTIVE MODE")
        print("="*50)
        print("Type 'quit' to exit\n")
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            intent, confidence, all_probs = predict_intent(user_input, model, vectorizer, label_encoder)
            response = get_response(intent, responses)
            
            print(f"\n[Intent: {intent} | Confidence: {confidence*100:.2f}%]")
            print(f"Bot: {response}\n")


if __name__ == "__main__":
    main()
