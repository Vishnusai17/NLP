"""
Intent Classifier Neural Network Model
A simple feedforward neural network for intent classification
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_model(input_dim, num_classes):
    """
    Create a feedforward neural network for intent classification.
    
    Args:
        input_dim (int): Size of the input vector (TF-IDF vocabulary size)
        num_classes (int): Number of intent classes to predict
    
    Returns:
        keras.Model: Compiled neural network model
    """
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(input_dim,)),
        
        # First hidden layer with 128 neurons
        layers.Dense(128, activation='relu', name='hidden_1'),
        layers.Dropout(0.5, name='dropout_1'),
        
        # Second hidden layer with 64 neurons
        layers.Dense(64, activation='relu', name='hidden_2'),
        layers.Dropout(0.5, name='dropout_2'),
        
        # Output layer with softmax for multi-class classification
        layers.Dense(num_classes, activation='softmax', name='output')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def get_model_summary(model):
    """
    Get a string representation of the model architecture.
    
    Args:
        model (keras.Model): The model to summarize
    
    Returns:
        str: Model summary as a string
    """
    summary_list = []
    model.summary(print_fn=lambda x: summary_list.append(x))
    return '\n'.join(summary_list)


if __name__ == "__main__":
    # Example usage
    print("Creating example model with input_dim=1000 and num_classes=7")
    example_model = create_model(input_dim=1000, num_classes=7)
    print("\n" + get_model_summary(example_model))
