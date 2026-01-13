"""
Movie Critic Chatbot - Generative LSTM Model
============================================
This script builds an LSTM model trained on IMDb movie reviews to generate
text completions in the style of movie critics.

Author: ML Expert
Date: 2026-01-12
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow_datasets as tfds
import re
import pickle
import os


class MovieCriticChatbot:
    """
    A generative LSTM chatbot that learns from IMDb movie reviews
    and generates text completions in the style of movie critics.
    """
    
    def __init__(self, vocab_size=3000, embedding_dim=32, max_sequence_len=30):
        """
        Initialize the chatbot with configuration parameters.
        
        Args:
            vocab_size: Maximum number of words in vocabulary
            embedding_dim: Dimension of word embeddings
            max_sequence_len: Maximum length of input sequences
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_sequence_len = max_sequence_len
        self.tokenizer = None
        self.model = None
        
    def preprocess_text(self, text):
        """
        Clean and preprocess text data.
        
        Args:
            text: Raw text string (bytes or str)
            
        Returns:
            Cleaned text string
        """
        # Convert bytes to string if needed
        if isinstance(text, bytes):
            text = text.decode('utf-8')
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Keep only letters, numbers, and basic punctuation
        text = re.sub(r'[^a-z0-9\s\.\,\!\?\'\-]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def load_and_prepare_data(self, num_samples=None):
        """
        Load IMDb reviews dataset and prepare training sequences.
        
        Args:
            num_samples: Limit number of samples (None = all data)
            
        Returns:
            X: Input sequences
            y: Target words (next word to predict)
        """
        print("Loading IMDb reviews dataset...")
        
        # Load the dataset automatically
        dataset, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)
        train_data, test_data = dataset['train'], dataset['test']
        
        # Extract and preprocess reviews
        reviews = []
        print("Preprocessing reviews...")
        
        for text, label in train_data:
            cleaned_text = self.preprocess_text(text.numpy())
            reviews.append(cleaned_text)
            
            if num_samples and len(reviews) >= num_samples:
                break
        
        print(f"Loaded {len(reviews)} reviews")
        
        # Tokenize the text
        print("Tokenizing text...")
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(reviews)
        total_words = min(len(self.tokenizer.word_index) + 1, self.vocab_size)
        print(f"Vocabulary size: {total_words}")
        
        # Create N-gram sequences for next word prediction
        print("Creating N-gram sequences...")
        input_sequences = []
        
        for review in reviews:
            token_list = self.tokenizer.texts_to_sequences([review])[0]
            
            # Limit sequence length to prevent memory issues
            if len(token_list) > 100:
                token_list = token_list[:100]
            
            # Create sequences of varying lengths
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i+1]
                input_sequences.append(n_gram_sequence)
                
                # Limit total sequences to prevent memory overflow
                if len(input_sequences) >= 300000:
                    break
            
            if len(input_sequences) >= 300000:
                print(f"Limiting to 300,000 sequences to conserve memory")
                break
        
        # Find the maximum sequence length
        max_len = max([len(seq) for seq in input_sequences])
        self.max_sequence_len = min(max_len, self.max_sequence_len)
        print(f"Max sequence length: {self.max_sequence_len}")
        
        # Pad sequences - using 'pre' padding to retain recent word memory
        # This is crucial for LSTM to remember the most recent context
        input_sequences = pad_sequences(
            input_sequences, 
            maxlen=self.max_sequence_len, 
            padding='pre'
        )
        
        # Split into input (X) and target (y)
        # X = all words except the last, y = the last word (what we want to predict)
        X = input_sequences[:, :-1]
        y = input_sequences[:, -1]
        
        # Convert y to categorical (one-hot encoding)
        y = tf.keras.utils.to_categorical(y, num_classes=total_words)
        
        print(f"Created {len(X)} training sequences")
        print(f"Input shape: {X.shape}, Output shape: {y.shape}")
        
        return X, y, total_words
    
    def build_model(self, total_words):
        """
        Build the LSTM model architecture.
        
        The architecture consists of:
        1. Embedding Layer: Converts word indices to dense vectors
        2. Bidirectional LSTM: Captures context from both directions
        3. LSTM: Additional layer for deeper learning
        4. Dropout: Prevents overfitting
        5. Dense: Output layer with softmax for next word prediction
        
        Args:
            total_words: Total vocabulary size
        """
        print("\nBuilding LSTM model...")
        
        model = Sequential([
            # Embedding Layer: Maps each word to a dense vector of embedding_dim
            # This learns semantic relationships between words
            Embedding(
                input_dim=total_words,
                output_dim=self.embedding_dim,
                input_length=self.max_sequence_len - 1
            ),
            
            # Bidirectional LSTM: Processes sequence in both forward and backward directions
            # MAXIMIZED for TPU v4 - Large capacity for better learning
            # return_sequences=True passes output to next LSTM layer
            Bidirectional(LSTM(200, return_sequences=True)),
            
            # Dropout: Randomly drops 20% of connections during training
            # This prevents overfitting and improves generalization
            Dropout(0.2),
            
            # Second LSTM Layer: Further processes the sequential information
            # LARGER for high-quality text generation
            LSTM(150),
            
            # Another Dropout layer for regularization
            Dropout(0.2),
            
            # Dense Layer: Outputs probability distribution over all words
            # Softmax ensures probabilities sum to 1
            Dense(total_words, activation='softmax')
        ])
        
        # Compile with categorical crossentropy (multi-class classification)
        # Adam optimizer adapts learning rate during training
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        
        self.model = model
        print(model.summary())
        
        return model
    
    def train(self, X, y, epochs=30, batch_size=64, validation_split=0.1):
        """
        Train the LSTM model.
        
        Args:
            X: Input sequences
            y: Target words
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data for validation
        """
        print("\nTraining the model...")
        
        # Callbacks for better training
        callbacks = [
            # Stop training if validation loss doesn't improve for 5 epochs
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            
            # Save the best model
            ModelCheckpoint(
                'best_movie_critic_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train the model
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def generate_text(self, seed_text, next_words=20, temperature=1.0):
        """
        Generate text completion based on seed text.
        
        Temperature controls randomness:
        - temperature < 1.0: More conservative, predictable
        - temperature = 1.0: Standard sampling
        - temperature > 1.0: More creative, diverse
        
        Args:
            seed_text: Starting text/phrase
            next_words: Number of words to generate
            temperature: Sampling temperature for creativity
            
        Returns:
            Generated text completion
        """
        generated_text = seed_text
        
        for _ in range(next_words):
            # Tokenize the current text
            token_list = self.tokenizer.texts_to_sequences([generated_text])[0]
            
            # Pad to match model input size (using 'pre' padding)
            token_list = pad_sequences(
                [token_list],
                maxlen=self.max_sequence_len - 1,
                padding='pre'
            )
            
            # Predict next word probabilities
            predictions = self.model.predict(token_list, verbose=0)[0]
            
            # Mask the <OOV> token (usually index 1) so it's never picked
            if 1 in self.tokenizer.index_word:
                predictions[1] = 0
                
            # Renormalize
            predictions = predictions / np.sum(predictions)
            
            # Apply temperature to predictions for controlled randomness
            predictions = np.log(predictions + 1e-10) / temperature
            predictions = np.exp(predictions) / np.sum(np.exp(predictions))
            
            # Sample next word based on probability distribution
            try:
                predicted_index = np.random.choice(len(predictions), p=predictions)
            except ValueError:
                # Fallback if probs sum to slightly > 1 or < 1 due to float precision
                predicted_index = np.argmax(predictions)
            
            # Convert index back to word
            output_word = ""
            for word, index in self.tokenizer.word_index.items():
                if index == predicted_index:
                    output_word = word
                    break
            
            if output_word:
                generated_text += " " + output_word
            else:
                break
        
        return generated_text
    
    def save_model(self, model_path='movie_critic_model.h5', tokenizer_path='tokenizer.pickle'):
        """Save the trained model and tokenizer."""
        self.model.save(model_path)
        
        with open(tokenizer_path, 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Save configuration
        config = {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'max_sequence_len': self.max_sequence_len
        }
        
        with open('model_config.pickle', 'wb') as handle:
            pickle.dump(config, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"\nModel saved to {model_path}")
        print(f"Tokenizer saved to {tokenizer_path}")
    
    def load_model(self, model_path='movie_critic_model.h5', tokenizer_path='tokenizer.pickle'):
        """Load a pre-trained model and tokenizer."""
        self.model = keras.models.load_model(model_path)
        
        with open(tokenizer_path, 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        
        with open('model_config.pickle', 'rb') as handle:
            config = pickle.load(handle)
            self.vocab_size = config['vocab_size']
            self.embedding_dim = config['embedding_dim']
            self.max_sequence_len = config['max_sequence_len']
        
        print(f"Model loaded from {model_path}")
    
    def interactive_chat(self):
        """
        Interactive chatbot loop for real-time text generation.
        Users can input seed phrases and get movie review completions.
        """
        print("\n" + "="*60)
        print("🎬 Movie Critic Chatbot - Interactive Mode 🎬")
        print("="*60)
        print("\nType a starting phrase (e.g., 'The acting was') and I'll")
        print("complete it in the style of a movie critic!")
        print("\nCommands:")
        print("  - Type 'quit' or 'exit' to stop")
        print("  - Type 'temp: X' to set temperature (e.g., 'temp: 1.5')")
        print("="*60 + "\n")
        
        temperature = 1.0
        
        while True:
            try:
                user_input = input("\n🎬 Your seed phrase: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Thanks for chatting! Goodbye!")
                    break
                
                # Check if user wants to change temperature
                if user_input.lower().startswith('temp:'):
                    try:
                        temperature = float(user_input.split(':')[1].strip())
                        temperature = max(0.1, min(2.0, temperature))  # Clamp between 0.1 and 2.0
                        print(f"✓ Temperature set to {temperature}")
                        continue
                    except:
                        print("❌ Invalid temperature. Use format: temp: 1.0")
                        continue
                
                # Generate text completion
                print("\n💭 Generating review...", end='', flush=True)
                generated = self.generate_text(
                    user_input,
                    next_words=30,
                    temperature=temperature
                )
                
                print("\r" + " "*50 + "\r", end='')  # Clear the "Generating..." message
                print(f"🎭 Generated Review:\n")
                print(f"   {generated}")
                print("\n" + "-"*60)
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                continue


def main():
    """Main execution function."""
    print("="*60)
    print("🎬 Movie Critic LSTM Chatbot")
    print("="*60)
    
    # Initialize chatbot (Balanced for Coherence)
    chatbot = MovieCriticChatbot(
        vocab_size=4000,
        embedding_dim=128,
        max_sequence_len=50
    )
    
    # Check if pre-trained model exists
    if os.path.exists('movie_critic_model.h5') and os.path.exists('tokenizer.pickle'):
        print("\n📂 Found existing model. Loading...")
        try:
            chatbot.load_model()
            print("✓ Model loaded successfully!")
            
            # Jump straight to interactive mode
            chatbot.interactive_chat()
            return
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("🔄 Training a new model instead...\n")
    
    # Load and prepare data
    # Using 3000 samples with lower vocab ensures higher word frequency overlap = better sentences
    X, y, total_words = chatbot.load_and_prepare_data(num_samples=3000)
    
    # Build model
    chatbot.build_model(total_words)
    
    # Train model
    # TPU v4 OPTIMIZED: Large batch size + more epochs = better quality
    history = chatbot.train(X, y, epochs=100, batch_size=256)
    
    # Save the trained model
    chatbot.save_model()
    
    print("\n" + "="*60)
    print("✓ Training Complete!")
    print("="*60)
    
    # Test with some examples
    print("\n📝 Sample Generations:\n")
    test_seeds = [
        "The acting was",
        "This movie is",
        "The plot was",
        "I really enjoyed"
    ]
    
    for seed in test_seeds:
        generated = chatbot.generate_text(seed, next_words=20)
        print(f"Seed: '{seed}'")
        print(f"Generated: {generated}\n")
    
    # Start interactive mode
    chatbot.interactive_chat()


if __name__ == "__main__":
    main()
