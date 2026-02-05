"""
Training Script for Intent Classifier
Trains a feedforward neural network on intent classification data
"""

import json
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from model import create_model


def load_training_data(filepath='training_data.json'):
    """
    Load training data from JSON file.
    
    Args:
        filepath (str): Path to the training data JSON file
    
    Returns:
        tuple: (patterns, labels) - lists of text patterns and their intent labels
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    patterns = []
    labels = []
    
    for intent in data['intents']:
        tag = intent['tag']
        for pattern in intent['patterns']:
            patterns.append(pattern.lower())  # Normalize to lowercase
            labels.append(tag)
    
    return patterns, labels


def preprocess_data(patterns, labels):
    """
    Preprocess text data using TF-IDF vectorization.
    
    Args:
        patterns (list): List of text patterns
        labels (list): List of intent labels
    
    Returns:
        tuple: (X, y, vectorizer, label_encoder)
    """
    # Convert text to TF-IDF vectors
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(patterns).toarray()
    
    # Encode labels as integers
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    
    return X, y, vectorizer, label_encoder


def train_model(X_train, y_train, X_val, y_val, num_classes, epochs=100):
    """
    Train the intent classification model.
    
    Args:
        X_train (np.array): Training features
        y_train (np.array): Training labels
        X_val (np.array): Validation features
        y_val (np.array): Validation labels
        num_classes (int): Number of intent classes
        epochs (int): Number of training epochs
    
    Returns:
        keras.Model: Trained model
    """
    input_dim = X_train.shape[1]
    model = create_model(input_dim, num_classes)
    
    print("\n" + "="*50)
    print("MODEL ARCHITECTURE")
    print("="*50)
    model.summary()
    
    print("\n" + "="*50)
    print("TRAINING MODEL")
    print("="*50)
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=8,
        verbose=1
    )
    
    return model, history


def save_artifacts(model, vectorizer, label_encoder):
    """
    Save the trained model and preprocessing artifacts.
    
    Args:
        model (keras.Model): Trained model
        vectorizer (TfidfVectorizer): Fitted vectorizer
        label_encoder (LabelEncoder): Fitted label encoder
    """
    # Save the model
    model.save('intent_classifier_model.h5')
    print("\n✓ Model saved to 'intent_classifier_model.h5'")
    
    # Save the vectorizer
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    print("✓ Vectorizer saved to 'vectorizer.pkl'")
    
    # Save the label encoder
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    print("✓ Label encoder saved to 'label_encoder.pkl'")


def main():
    print("="*50)
    print("INTENT CLASSIFIER TRAINING")
    print("="*50)
    
    # Load data
    print("\n1. Loading training data...")
    patterns, labels = load_training_data()
    print(f"   Loaded {len(patterns)} patterns across {len(set(labels))} intents")
    
    # Preprocess data
    print("\n2. Preprocessing data...")
    X, y, vectorizer, label_encoder = preprocess_data(patterns, labels)
    print(f"   Feature vector size: {X.shape[1]}")
    print(f"   Intent classes: {list(label_encoder.classes_)}")
    
    # Split data
    print("\n3. Splitting data into train/validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Training samples: {len(X_train)}")
    print(f"   Validation samples: {len(X_val)}")
    
    # Train model
    print("\n4. Creating and training model...")
    num_classes = len(label_encoder.classes_)
    model, history = train_model(X_train, y_train, X_val, y_val, num_classes, epochs=100)
    
    # Evaluate
    print("\n" + "="*50)
    print("FINAL EVALUATION")
    print("="*50)
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Training Accuracy: {train_acc*100:.2f}%")
    print(f"Validation Accuracy: {val_acc*100:.2f}%")
    
    # Save artifacts
    print("\n5. Saving model and artifacts...")
    save_artifacts(model, vectorizer, label_encoder)
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)
    print("\nYou can now use 'predict.py' to classify intents!")


if __name__ == "__main__":
    main()
