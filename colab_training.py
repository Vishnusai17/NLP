"""
Movie Critic LSTM - Google Colab Training Script
================================================
This script is optimized for Google Colab with GPU acceleration.
It trains a high-quality model and saves it for download.

Instructions:
1. Upload this file to Google Colab
2. Runtime → Change runtime type → GPU (T4)
3. Run all cells
4. Download the generated model files at the end
5. Place them in your local project folder
"""

# ============================================================
# STEP 1: Install Dependencies
# ============================================================
print("📦 Installing dependencies...")
!pip install -q tensorflow tensorflow-datasets

# ============================================================
# STEP 2: Import Libraries
# ============================================================
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

print(f"✓ TensorFlow version: {tf.__version__}")
print(f"✓ GPU available: {tf.config.list_physical_devices('GPU')}")

# ============================================================
# STEP 3: MovieCriticChatbot Class
# ============================================================
class MovieCriticChatbot:
    """
    A generative LSTM chatbot trained on IMDb movie reviews.
    Optimized for Google Colab training.
    """
    
    def __init__(self, vocab_size=10000, embedding_dim=128, max_sequence_len=50):
        """Initialize with HIGH-QUALITY settings for Colab."""
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_sequence_len = max_sequence_len
        self.tokenizer = None
        self.model = None
        
    def preprocess_text(self, text):
        """Clean and preprocess text data."""
        if isinstance(text, bytes):
            text = text.decode('utf-8')
        text = text.lower()
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'[^a-z0-9\s\.\,\!\?\'\-]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def load_and_prepare_data(self, num_samples=10000):
        """
        Load IMDb reviews and prepare training sequences.
        Using MORE data for better quality on Colab.
        """
        print("\n" + "="*60)
        print("📊 LOADING IMDB DATASET")
        print("="*60)
        
        # Load dataset
        dataset, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)
        train_data = dataset['train']
        
        # Extract and preprocess reviews
        reviews = []
        print(f"Processing {num_samples} reviews...")
        
        for i, (text, label) in enumerate(train_data):
            cleaned_text = self.preprocess_text(text.numpy())
            reviews.append(cleaned_text)
            
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{num_samples} reviews...")
            
            if len(reviews) >= num_samples:
                break
        
        print(f"✓ Loaded {len(reviews)} reviews\n")
        
        # Tokenize
        print("🔤 Tokenizing text...")
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(reviews)
        total_words = min(len(self.tokenizer.word_index) + 1, self.vocab_size)
        print(f"✓ Vocabulary size: {total_words}\n")
        
        # Create N-gram sequences
        print("🔢 Creating N-gram sequences...")
        input_sequences = []
        
        for idx, review in enumerate(reviews):
            token_list = self.tokenizer.texts_to_sequences([review])[0]
            
            # Limit token length to prevent extreme memory usage
            if len(token_list) > 200:
                token_list = token_list[:200]
            
            # Create sequences
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i+1]
                input_sequences.append(n_gram_sequence)
                
                # Safety limit for sequences (can be higher on Colab)
                if len(input_sequences) >= 200000:
                    print(f"⚠️  Limiting to 200,000 sequences for memory efficiency")
                    break
            
            if len(input_sequences) >= 200000:
                break
            
            if (idx + 1) % 1000 == 0:
                print(f"  Created {len(input_sequences)} sequences from {idx + 1} reviews...")
        
        # Find max sequence length
        max_len = max([len(seq) for seq in input_sequences])
        self.max_sequence_len = min(max_len, self.max_sequence_len)
        print(f"✓ Max sequence length: {self.max_sequence_len}")
        
        # Pad sequences with 'pre' padding
        print("📏 Padding sequences...")
        input_sequences = pad_sequences(
            input_sequences, 
            maxlen=self.max_sequence_len, 
            padding='pre'
        )
        
        # Split into X and y
        X = input_sequences[:, :-1]
        y = input_sequences[:, -1]
        y = tf.keras.utils.to_categorical(y, num_classes=total_words)
        
        print(f"✓ Created {len(X)} training sequences")
        print(f"✓ Input shape: {X.shape}, Output shape: {y.shape}\n")
        
        return X, y, total_words
    
    def build_model(self, total_words):
        """
        Build HIGH-QUALITY LSTM model architecture.
        Larger than the local version since Colab has more resources.
        """
        print("="*60)
        print("🧠 BUILDING LSTM MODEL")
        print("="*60 + "\n")
        
        model = Sequential([
            # Embedding Layer
            Embedding(
                input_dim=total_words,
                output_dim=self.embedding_dim,
                input_length=self.max_sequence_len - 1
            ),
            
            # Bidirectional LSTM - LARGER for better quality
            Bidirectional(LSTM(150, return_sequences=True)),
            Dropout(0.2),
            
            # Second LSTM Layer
            LSTM(100),
            Dropout(0.2),
            
            # Dense output layer
            Dense(total_words, activation='softmax')
        ])
        
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        
        self.model = model
        print(model.summary())
        print()
        
        return model
    
    def train(self, X, y, epochs=50, batch_size=128):
        """Train the model with GPU acceleration."""
        print("="*60)
        print("🚀 TRAINING MODEL (GPU Accelerated)")
        print("="*60 + "\n")
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss', 
                patience=5, 
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                'best_movie_critic_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def save_model(self):
        """Save model and tokenizer for download."""
        print("\n" + "="*60)
        print("💾 SAVING MODEL")
        print("="*60)
        
        # Save model
        self.model.save('movie_critic_model.h5')
        print("✓ Model saved to: movie_critic_model.h5")
        
        # Save tokenizer
        with open('tokenizer.pickle', 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("✓ Tokenizer saved to: tokenizer.pickle")
        
        # Save config
        config = {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'max_sequence_len': self.max_sequence_len
        }
        with open('model_config.pickle', 'wb') as handle:
            pickle.dump(config, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("✓ Config saved to: model_config.pickle")
        
        print("\n📥 DOWNLOAD THESE 3 FILES:")
        print("   1. movie_critic_model.h5")
        print("   2. tokenizer.pickle")
        print("   3. model_config.pickle")
        print("\nPlace them in your local project folder!\n")
    
    def generate_text(self, seed_text, next_words=30, temperature=1.0):
        """Generate text completion."""
        generated_text = seed_text
        
        for _ in range(next_words):
            token_list = self.tokenizer.texts_to_sequences([generated_text])[0]
            token_list = pad_sequences(
                [token_list],
                maxlen=self.max_sequence_len - 1,
                padding='pre'
            )
            
            predictions = self.model.predict(token_list, verbose=0)[0]
            predictions = np.log(predictions + 1e-10) / temperature
            predictions = np.exp(predictions) / np.sum(np.exp(predictions))
            
            predicted_index = np.random.choice(len(predictions), p=predictions)
            
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


# ============================================================
# STEP 4: MAIN TRAINING PIPELINE
# ============================================================
def main():
    print("\n" + "="*60)
    print("🎬 MOVIE CRITIC LSTM - COLAB TRAINING")
    print("="*60 + "\n")
    
    # Initialize with HIGH-QUALITY settings
    chatbot = MovieCriticChatbot(
        vocab_size=10000,      # Large vocabulary
        embedding_dim=128,     # Rich embeddings
        max_sequence_len=50    # Longer sequences
    )
    
    # Load data - Using 10,000 reviews for high quality
    # Reduce num_samples if you encounter memory issues
    X, y, total_words = chatbot.load_and_prepare_data(num_samples=10000)
    
    # Build model
    chatbot.build_model(total_words)
    
    # Train model - 50 epochs with early stopping
    history = chatbot.train(X, y, epochs=50, batch_size=128)
    
    # Save model
    chatbot.save_model()
    
    # Test generation
    print("="*60)
    print("🎭 TESTING TEXT GENERATION")
    print("="*60 + "\n")
    
    test_seeds = [
        "The acting was",
        "This movie is",
        "The plot was",
        "I really enjoyed"
    ]
    
    for seed in test_seeds:
        generated = chatbot.generate_text(seed, next_words=25, temperature=1.0)
        print(f"Seed: '{seed}'")
        print(f"Generated: {generated}\n")
    
    print("="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print("\n📥 NOW DOWNLOAD THESE FILES:")
    print("   • movie_critic_model.h5")
    print("   • tokenizer.pickle")
    print("   • model_config.pickle")
    print("\n📂 Place them in your local LSTM folder")
    print("🌐 Then run: python app.py")
    print("\n" + "="*60 + "\n")


# ============================================================
# RUN TRAINING
# ============================================================
if __name__ == "__main__":
    main()
