# Movie Critic Chatbot 🎬

![Movie Critic AI Interface](static/demo.png)

An intelligent AI chatbot that generates movie reviews using an LSTM (Long Short-Term Memory) neural network trained on the IMDb dataset.

## ✨ Features

- **🧠 Advanced LSTM**: Stacked Bidirectional architecture optimized for coherence.
- **⚡ TPU/GPU Ready**: Optimized for high-performance training on TPU v4 or Google Colab.
- **🎨 Modern Web UI**: Beautiful Flask-based interface with glassmorphism design.
- **💬 Real-time Generation**: Instant review completion with temperature control.
- **🛡️ Robust output**: Filters "unknown" tokens for cleaner text generation.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Choose Your Training Method

#### Option A: High-Quality (Recommended) 🌟
Use Google Colab to train a professional-grade model (10k reviews).
1. Open `colab_training.py` in [Google Colab](https://colab.research.google.com/).
2. Enable GPU (Runtime -> Change runtime type -> GPU).
3. Run all cells.
4. Download `movie_critic_model.h5`, `tokenizer.pickle`, and `model_config.pickle`.
5. Place files in this directory.

#### Option B: Fast Local Training ⚡
Train a lighter model on your machine (3k reviews).
```bash
python movie_critic_chatbot.py
```

### 3. Run the Chatbot
Start the web server:
```bash
python app.py
```
Open your browser to: **http://localhost:5001**

## 🎮 How to Use

1. **Enter a seed phrase**: e.g., "The acting was", "I really loved".
2. **Adjust settings**:
   - **Temperature**: Higher (1.2+) = Creative/Wild, Lower (0.8) = Safe/Predictable.
   - **Word Count**: Choose length of the review.
3. **Chat**: The AI will complete your sentence in the style of a movie critic!

## 📂 Project Structure

- `app.py`: Flask web server and API.
- `movie_critic_chatbot.py`: Core LSTM model definition and training logic.
- `colab_training.py`: Script optimized for Google Colab.
- `templates/index.html`: Modern web interface.
- `static/`: CSS and JS assets.

## 🔧 Troubleshooting

- **Port in use?**: The app runs on port 5001 to avoid AirPlay conflicts on macOS.
- **Garbage text?**: Ensure you are using the Colab model or have retrained locally with the latest "coherence" settings.
